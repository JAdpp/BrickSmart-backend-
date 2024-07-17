import json
import os
import time
from .models import ModelTask, ComponentList
from .forms import PromptForm, ImageUploadForm
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import requests
import re
import django.http.request
from BackEnd.settings import BASE_DIR, MEDIA_ROOT
import threading
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import qiniu

# model.py
# @csrf_exempt
# 视图函数，用于文生模型
# @csrf_exempt
# def generate_model(request):
#     # 检查请求方法
#     if request.method == 'POST':
#         # 创建并验证表单
#         form = PromptForm(request.POST)
#         if form.is_valid():
#             # 提取用户输入的提示信息
#             prompt = form.cleaned_data['prompt']
#
#             # 调用API
#             api_url = 'https://api.tripo3d.ai/v2/openapi/task'
#             api_key = ""
#             # 构建请求头，包含API密钥
#             header = {
#                 'Authorization': f'Bearer {api_key}'
#             }
#
#             # 发送POST请求到API
#             response = requests.post(api_url, json={'type': 'text_to_model', 'prompt': prompt}, headers=header)
#             # 解析API响应为JSON
#             create_response_data = response.json()
#             # 提取任务ID
#             task_id = create_response_data['data']['task_id']
#             print(f"Task ID: {task_id}")
#
#             # 构建获取任务状态的URL
#             get_task_url = f"https://api.tripo3d.ai/v2/openapi/task/{task_id}"
#             # 构建获取任务状态的请求头
#             get_headers = {
#                 'Authorization': f'Bearer {api_key}'
#             }
#
#             # 等待一段时间，确保任务处理完成
#             # 实际等待时间可能需要根据任务的复杂度调整
#             time.sleep(25)
#
#             # 发送GET请求到API获取任务状态
#             get_response = requests.get(get_task_url, headers=get_headers)
#             # 解析获取的API响应为JSON
#             get_response_data = get_response.json()
#
#             # 检查任务状态是否为成功
#             if get_response_data['data']['status'] == 'success':
#                 # 提取模型URL和渲染图像URL
#                 model_url = get_response_data['data']['output']['model']
#                 image_url = get_response_data['data']['output']['rendered_image']
#
#                 # 启动多线程下载并保存文件
#                 threading.Thread(target=download_and_save_file, args=(model_url, '3D-model', task_id, 'glb')).start()
#                 threading.Thread(target=download_and_save_file, args=(image_url, 'rendered-image', task_id, 'webp')).start()
#
#                 # 保存任务数据到数据库
#                 ModelTask.objects.create(
#                     task_id=task_id,
#                     prompt=prompt,
#                     bricks="default",  # Replace with actual value as needed
#                     model_download_url= f"3D-model/{task_id}.glb",
#                     image_download_url= f"rendered-image/{task_id}.webp",
#                     lego_url="",  # Assuming this gets set elsewhere
#                     user=request.user
#                 )
#
#                 # 返回渲染后的HTML页面，包含模型URL
#                 return render(request, 'model.html', {'model_url': model_url})
#     else:
#         # 如果请求方法不是POST，创建并返回提示页面和表单
#         form = PromptForm()
#
#     # 返回提示页面
#     return render(request, 'prompt.html', {'form': form})


#七牛云
access_key = 'WF88Hagl_Oev5A7qj8Dp0bLdkPCJq9PPISNfpABN'
secret_key = 'TJobSkP14PD5Qn9QGab0g8bIQeLBn9TDBbncjvXw'

# 构建鉴权对象
q = qiniu.Auth(access_key, secret_key)

# 要上传的空间
bucket_name = 'stay33'

#七牛云上传
def upload_file_to_qiniu(local_file_path, key):
    # 生成上传 Token，可以指定过期时间等
    token = q.upload_token(bucket_name, key, 3600)

    # 构建一个上传对象
    ret, info = qiniu.put_file(token, key, local_file_path)

    if info.status_code == 200:
        print(f"Upload success! URL: http://your_domain/{key}")
        return f"https://qiniu.staykoi.asia/{key}"
    else:
        print("Upload failed!")
        print(info)
        return None

#下载保存文件
def download_and_save_file(url, folder, task_id, extension):
    response = requests.get(url)
    if response.status_code == 200:
        file_path = f"{BASE_DIR.parent}/{folder}/{task_id}.{extension}"
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File saved to {file_path}")

        # 上传到七牛云
        key = f"{folder}/{task_id}.{extension}"
        qiniu_url = upload_file_to_qiniu(file_path, key)
        return qiniu_url
    else:
        print(f"Failed to download file from {url}")
        return None
    
#文生3D
@csrf_exempt
def generate_model(request):
    # 获取所有元件
    components = ComponentList.objects.all() 
    if request.method == 'POST':
      #  try:
      #      request_data = json.loads(request.body)
      #      prompt = request_data.get('prompt', None)
      #  except json.JSONDecodeError:
      #      return JsonResponse({'error': 'Invalid JSON'}, status=400)

      #  if not prompt:
      #      return JsonResponse({'error': 'This field is required.'}, status=400)

        # 创建并验证表单
        form = PromptForm(request.POST)
        if form.is_valid():
            prompt = form.cleaned_data['prompt']
            
            # 构建元件信息列表
            selected_components = []
            for component in components:
                try:
                    component_qty = int(request.POST.get(f'component_qty_{component.pid}', 0))
                    if int(component_qty) > 0:
                        selected_components.append({'sn': component.sn, 'qty': component_qty})
                except ValueError:
                    continue  # 如果转换失败，跳过这个元件
            # JSON化元件信息
            bricks_json = json.dumps(selected_components)

            api_url = 'https://api.tripo3d.ai/v2/openapi/task'
            api_key = "tsk_yZUd8eTShQwltj-XZY8Et4P3WWBWNPouEg1O23OzfS0"  # 设置你的API密钥
            header = {
                'Authorization': f'Bearer {api_key}'
            }

            response = requests.post(api_url, json={'type': 'text_to_model', 'prompt': prompt}, headers=header)
            if response.status_code != 200:
                return JsonResponse({'error': 'API request failed', 'details': response.text}, status=response.status_code)

            create_response_data = response.json()
            if 'data' not in create_response_data:
                return JsonResponse({'error': 'API response error', 'details': create_response_data}, status=500)

            task_id = create_response_data['data']['task_id']
            print(f"Task ID: {task_id}")

            get_task_url = f"https://api.tripo3d.ai/v2/openapi/task/{task_id}"
            get_headers = {
                'Authorization': f'Bearer {api_key}'
            }

            time.sleep(25)

            get_response = requests.get(get_task_url, headers=get_headers)
            if get_response.status_code != 200:
                return JsonResponse({'error': 'API request failed', 'details': get_response.text}, status=get_response.status_code)

            get_response_data = get_response.json()
            if 'data' not in get_response_data or get_response_data['data'].get('status') != 'success':
                return JsonResponse({'error': 'API response error', 'details': get_response_data}, status=500)

            model_url = get_response_data['data']['output']['model']
            image_url = get_response_data['data']['output']['rendered_image']

            model_filename = f"{task_id}.glb"
            image_filename = f"{task_id}.webp"

            # 下载并上传到七牛云
            model_qiniu_url = download_and_save_file(model_url, '3D-model', task_id, 'glb')
            image_qiniu_url = download_and_save_file(image_url, 'rendered-image', task_id, 'webp')

            user = request.user if request.user.is_authenticated else None

            ModelTask.objects.create(
                 task_id=task_id,
                 prompt=prompt,
                 bricks=bricks_json, # 使用JSON字符串存储组件信息
                 model_download_url=f"3D-model/{task_id}.glb",
                 image_download_url=f"rendered-image/{task_id}.webp",
                 lego_url="",
                 user=user
             )

            return JsonResponse({'model_filename': model_qiniu_url, 'image_filename': image_qiniu_url})
        else:
            return JsonResponse({'error': 'Invalid form data'}, status=400)
    else:
        form = PromptForm()

    return render(request, 'prompt.html', {'form': form, 'components': components})

# 图生3D
@csrf_exempt # 使用csrf_exempt装饰器，允许跨站请求伪造保护被禁用
def generate_model_image(request):
    # 获取所有元件
    components = ComponentList.objects.all() 
    if request.method == 'POST':  # 判断请求方法是否为POST
        form = ImageUploadForm(request.POST, request.FILES)
        print(request.FILES) # 打印上传的文件信息

        if form.is_valid(): # 判断表单是否有效
            image = form.cleaned_data['image'] # 获取上传的图片
            image_path = default_storage.save('uploads/' + image.name, ContentFile(image.read())) # 将图片保存到指定路径

            # 构建元件信息列表
            selected_components = []
            for component in components:
                try:
                    component_qty = int(request.POST.get(f'component_qty_{component.pid}', 0))
                    if int(component_qty) > 0:
                        selected_components.append({'sn': component.sn, 'qty': component_qty})
                except ValueError:
                    continue  # 如果转换失败，跳过这个元件
            # JSON化元件信息
            bricks_json = json.dumps(selected_components)

            # 设置API的URL和密钥
            api_url = "https://api.tripo3d.ai/v2/openapi/upload"
            api_key = "tsk_S9zAZ08NFPuKt9le3qqr6rDbHUQO38dVoqm7zOf1U49"
            headers = {
                "Authorization": f"Bearer {api_key}"
            }

            # 构建文件路径和文件类型
            file_path = os.path.join(MEDIA_ROOT, image_path)
            files = {'file': (file_path, open(file_path, 'rb'), 'image/jpeg')}
            print(files)

            # 发送POST请求以上传图像并获取响应
            response = requests.post(api_url, headers=headers, files=files).json()
            print(response)

            if response['code'] == 0:
                file_token = response['data']['image_token']
                print(f"image token: {file_token}")
                create_task_url = "https://api.tripo3d.ai/v2/openapi/task"
                create_headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}'
                }
                create_data = {
                    'type': 'image_to_model',
                    "file": {
                        "type": "jpg",
                        "file_token": file_token
                    }
                }

                # 等待一段时间，确保图像上传任务处理完成
                # 实际等待时间可能需要根据任务的复杂度调整
                time.sleep(5)

                # 创建任务用于将图像转换为模型
                create_response = requests.post(create_task_url, json=create_data, headers=create_headers)
                create_response_data = create_response.json()

                if create_response_data['code'] == 0:
                    task_id = create_response_data['data']['task_id']
                    print(f"Task ID: {task_id}")

                    # 构建用于获取结果的 URL 和请求头
                    get_task_url = f"https://api.tripo3d.ai/v2/openapi/task/{task_id}"
                    get_headers = {
                        'Authorization': f'Bearer {api_key}'
                    }

                    # 等待一段时间，确保任务处理完成
                    time.sleep(25)  # 实际等待时间可能需要根据任务的复杂度调整

                    # 获取任务结果
                    get_response = requests.get(get_task_url, headers=get_headers)
                    get_response_data = get_response.json()
                    if get_response_data['data']['status'] == 'success':
                        model_url = get_response_data['data']['output']['model']
                        image_url = get_response_data['data']['output']['rendered_image']

                    # 使用线程异步下载并保存模型和渲染图像文件
                    threading.Thread(target=download_and_save_file,
                                        args=(model_url, '3D-model', task_id, 'glb')).start()
                    threading.Thread(target=download_and_save_file,
                                        args=(image_url, 'rendered-image', task_id, 'webp')).start()

                    # 保存任务数据到数据库
                    ModelTask.objects.create(
                        task_id=task_id,
                        prompt=image_path,
                        bricks=bricks_json,  
                        model_download_url= f"3D-model/{task_id}.glb",
                        image_download_url= f"rendered-image/{task_id}.webp",
                        lego_url="",  # Assuming this gets set elsewhere
                        user=request.user
                    )

                    # 返回模型页面并展示模型URL
                    return render(request, 'model.html', {'model_url': model_url})
    else:
        form = ImageUploadForm()

    # 返回上传页面或者表单错误信息
    return render(request, 'upload.html', {'form': form, 'components': components})


# @csrf_exempt
# def generate_model_textImage(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             messages = data.get('messages', [])
#             model = data.get('model', '')  # 获取model字段
#
#             if not messages:
#                 return JsonResponse({'error': 'Messages is required'}, status=400)
#             if not model:  # 检查model字段
#                 return JsonResponse({'error': 'Model is required'}, status=400)
#
#             api_url = 'https://api.gptgod.online/v1/chat/completions'
#             api_key = "sk-Mjo3suto8oJF4E9YfwyRxehDwRTD80RjhYRzYNfpXj3sFIV4"
#
#             headers = {
#                 'Authorization': f'Bearer {api_key}',
#                 'Content-Type': 'application/json'
#             }
#
#             request_data = {
#                 'messages': messages,
#                 'model': model  # 包含model字段
#             }
#
#             response = requests.post(api_url, json=request_data, headers=headers)
#             create_response_data = response.json()
#
#             # 打印响应状态码和内容以便调试
#             print("API response status code:", response.status_code)
#             print("API response content:", create_response_data)
#
#             if response.status_code == 200:
#                 choices = create_response_data['choices'][0]['message']['content']
#                 image_urls = re.findall(r'!\[image\d*\]\((https://[^\)]+)\)', choices)
#
#                 if not image_urls:
#                     return JsonResponse({'error': 'No valid image URLs found in the API response'}, status=500)
#
#                 print(f"Image URLs: {image_urls}")
#
#                 task_id = create_response_data['id']
#                 print(f"Task ID: {task_id}")
#
#                 # 下载并保存图片
#                 for idx, image_url in enumerate(image_urls):
#                     threading.Thread(target=download_and_save_file, args=(image_url, f'rendered-image-{idx}', task_id, 'webp')).start()
#
#                 return JsonResponse({'image_urls': image_urls})
#             else:
#                 error_message = create_response_data.get('message', 'Unknown error')
#                 return JsonResponse({'error': f'Failed to create task: {error_message}'}, status=500)
#         except json.JSONDecodeError:
#             return JsonResponse({'error': 'Invalid JSON format'}, status=400)
#         except Exception as e:
#             print(f"Unexpected error: {str(e)}")
#             return JsonResponse({'error': str(e)}, status=500)
#     else:
#         return JsonResponse({'error': 'Invalid request method'}, status=405)


# @csrf_exempt
# def generate_model_image(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         prompt = data.get('prompt', '')
#
#         # 设置API的URL和密钥
#         api_url = "https://api.tripo3d.ai/v2/openapi/task"
#         api_key = "tsk_S9zAZ08NFPuKt9le3qqr6rDbHUQO38dVoqm7zOf1U49"
#         headers = {
#             'Content-Type': 'application/json',
#             'Authorization': f'Bearer {api_key}'
#         }
#         create_data = {
#             'type': 'text_to_image',  # 假设有一个用于生成图像的API类型
#             'prompt': prompt
#         }
#
#         # 创建任务用于将文本转换为模型
#         create_response = requests.post(api_url, json=create_data, headers=headers)
#         create_response_data = create_response.json()
#
#         if create_response_data['code'] == 0:
#             task_id = create_response_data['data']['task_id']
#             print(f"Task ID: {task_id}")
#
#             # 构建用于获取结果的 URL 和请求头
#             get_task_url = f"https://api.tripo3d.ai/v2/openapi/task/{task_id}"
#             get_headers = {
#                 'Authorization': f'Bearer {api_key}'
#             }
#
#             # 等待一段时间，确保任务处理完成
#             time.sleep(25)  # 实际等待时间可能需要根据任务的复杂度调整
#
#             # 获取任务结果
#             get_response = requests.get(get_task_url, headers=get_headers)
#             get_response_data = get_response.json()
#             if get_response_data['data']['status'] == 'success':
#                 model_url = get_response_data['data']['output']['model']
#                 image_url = get_response_data['data']['output']['rendered_image']
#
#                 # 使用线程异步下载并保存模型和渲染图像文件
#                 threading.Thread(target=download_and_save_file,
#                                  args=(model_url, '3D-model', task_id, 'glb')).start()
#                 threading.Thread(target=download_and_save_file,
#                                  args=(image_url, 'rendered-image', task_id, 'webp')).start()
#
#                 ModelTask.objects.create(
#                     task_id=task_id,
#                     prompt=prompt,
#                     bricks="default",
#                     model_download_url=f"3D-model/{task_id}.glb",
#                     image_download_url=f"rendered-image/{task_id}.webp",
#                     lego_url="",
#                     user=request.user
#                 )
#
#                 return JsonResponse({'model_url': model_url, 'image_url': image_url})
#
#         return JsonResponse({'error': 'Failed to create task'}, status=500)
#     return JsonResponse({'error': 'Invalid request method'}, status=405)
# def download_and_save_file(url, folder, task_id, extension):
#     # 构建文件夹路径，确保它在 data 目录下
#     folder_path = os.path.join(BASE_DIR, 'data', folder)
#
#     # 检查目标文件夹是否存在，不存在则创建
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#
#     # 构建文件名和完整路径
#     file_name = f"{task_id}.{extension}"
#     file_path = os.path.join(folder_path, file_name)
#
#     # 请求文件内容
#     response = requests.get(url)
#     if response.status_code == 200:
#         # 构建完整的文件路径
#         file_path = os.path.join(folder_path, file_name)
#         # 写入文件
#         with open(file_path, 'wb') as file:
#             file.write(response.content)
#         print(f"File saved to {file_path}")
#     else:
#         print(f"Failed to download {file_name}, status code: {response.status_code}")