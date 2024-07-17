import json
import os
import time
from .models import ModelTask, ComponentList
from .forms import PromptForm, ImageUploadForm
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import requests
import django.http.request
from BackEnd.settings import BASE_DIR, MEDIA_ROOT
import threading
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


# views.py
@csrf_exempt
# 视图函数，用于文生3D模型
def generate_model(request):
    # 获取所有元件
    components = ComponentList.objects.all() 
    # 检查请求方法
    if request.method == 'POST':
        # 创建并验证表单
        form = PromptForm(request.POST)
        if form.is_valid():
            # 提取用户输入的提示信息
            prompt = form.cleaned_data['prompt']
            negative_prompt = "low quality, low resolution, low poly, ugly"
            
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

            # 调用API
            api_url = 'https://api.tripo3d.ai/v2/openapi/task'
            api_key = "tsk_t3rmrzT7v4Pow6t2D3jK1h8I1S0bCecUZ1EjTPz7jaI" #设置API秘钥
            # 构建请求头，包含API密钥
            header = {
                'Authorization': f'Bearer {api_key}'
            }
            
            # 发送POST请求到API
            response = requests.post(api_url, json={'type': 'text_to_model', 'prompt': prompt, 'negative_prompt': negative_prompt}, headers=header)
            # 解析API响应为JSON
            create_response_data = response.json()
            # 提取任务ID
            task_id = create_response_data['data']['task_id']
            print(f"Task ID: {task_id}")
            
            # 构建获取任务状态的URL
            get_task_url = f"https://api.tripo3d.ai/v2/openapi/task/{task_id}"
            # 构建获取任务状态的请求头
            get_headers = {
                'Authorization': f'Bearer {api_key}'
            }
            
            # 等待一段时间，确保任务处理完成
            # 实际等待时间可能需要根据任务的复杂度调整
            time.sleep(25)
            
            # 发送GET请求到API获取任务状态
            get_response = requests.get(get_task_url, headers=get_headers)
            # 解析获取的API响应为JSON
            get_response_data = get_response.json()
            
            # 检查任务状态是否为成功
            if get_response_data['data']['status'] == 'success':
                # 提取模型URL和渲染图像URL
                model_url = get_response_data['data']['output']['model']
                image_url = get_response_data['data']['output']['rendered_image']
                
                # 启动多线程下载并保存文件
                threading.Thread(target=download_and_save_file, args=(model_url, '3D-model', task_id, 'glb')).start()
                threading.Thread(target=download_and_save_file, args=(image_url, 'rendered-image', task_id, 'webp')).start()

                # 保存任务数据到数据库
                ModelTask.objects.create(
                    task_id=task_id,
                    prompt=prompt,
                    bricks=bricks_json,  # 使用JSON字符串存储组件信息
                    model_download_url= f"3D-model/{task_id}.glb",
                    image_download_url= f"rendered-image/{task_id}.webp",
                    lego_url="",  # Assuming this gets set elsewhere
                    user=request.user
                )
                
                # 返回渲染后的HTML页面，包含模型URL
                return render(request, 'model.html', {'model_url': model_url})
    else:
        # 如果请求方法不是POST，创建并返回提示页面和表单
        form = PromptForm()
    return render(request, 'prompt.html', {'form': form, 'components': components})


# 视图函数，用于图生3D模型
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
            api_key = "tsk_t3rmrzT7v4Pow6t2D3jK1h8I1S0bCecUZ1EjTPz7jaI"
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
                        bricks=bricks_json,  # 使用JSON字符串存储组件信息
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


def download_and_save_file(url, folder, task_id, extension):
    # 构建文件夹路径，确保它在 data 目录下
    folder_path = os.path.join(BASE_DIR, 'data', folder)

    # 检查目标文件夹是否存在，不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 构建文件名和完整路径
    file_name = f"{task_id}.{extension}"
    file_path = os.path.join(folder_path, file_name)

    # 请求文件内容
    response = requests.get(url)
    if response.status_code == 200:
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, file_name)
        # 写入文件
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File saved to {file_path}")
    else:
        print(f"Failed to download {file_name}, status code: {response.status_code}")
