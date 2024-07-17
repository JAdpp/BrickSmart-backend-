from django.db import models
from django.contrib.auth.models import User

# 文生3D模型任务表
class ModelTask(models.Model):
    task_id = models.CharField(max_length=255, unique=True, verbose_name='任务_id')
    prompt = models.TextField(verbose_name='提示词')
    bricks = models.TextField(verbose_name='选择的元件')  # 使用 Text 字段来存储 JSON 数据
    model_download_url = models.URLField(verbose_name='模型保存地址')
    image_download_url = models.URLField(verbose_name='模型预览图保存地址')
    lego_url = models.URLField(verbose_name='乐高方案保存地址')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_time = models.DateTimeField(auto_now_add=True,verbose_name='创建日期')

    def __str__(self):
        return self.task_id
    
    class Meta:
        verbose_name = '生成任务表'
        verbose_name_plural = verbose_name


# 元件表   
class ComponentList(models.Model):
    pid = models.AutoField(primary_key=True,verbose_name='元件_id')
    sn = models.CharField(max_length=64,verbose_name='元件编号')
    caption = models.CharField(max_length=64,verbose_name='元件名称')
    type_choices = (
        (0, '未定义'),
        (1, '条'),
        (2, '板'),
        (3, '块'),
        (4, '其他'),
    )
    btype = models.IntegerField(verbose_name='元件类型', choices=type_choices, default=0)
    color_choices = (
        (1, '白'),
        (2, '黑'),
        (3, '红'),
        (4, '绿'),
    )
    bcolor = models.IntegerField(verbose_name='元件颜色', choices=color_choices, default=0)
    package = models.ForeignKey(verbose_name='所属套装', to='PackageList',on_delete=models.PROTECT, default=1)
    imgFile = models.FileField(null=True,blank=True,upload_to='bricks/',verbose_name='元件img文件')
    createTime_Internal = models.DateTimeField(auto_now_add=True,verbose_name='创建日期')
    alterTime_Internal = models.DateTimeField(auto_now=True,verbose_name='修改日期')

    def __str__(self):
        return self.caption
    
    class Meta:
        verbose_name = '元件表'
        verbose_name_plural = verbose_name

# 元件套装表
class PackageList(models.Model):
    pid = models.AutoField(primary_key=True,verbose_name='套装_id')
    caption = models.CharField(max_length=64,verbose_name='套装名称')
    description = models.CharField(max_length=64,null=True,blank=True,verbose_name='套装描述')
    createTime_Internal = models.DateTimeField(auto_now_add=True,verbose_name='创建日期')
    alterTime_Internal = models.DateTimeField(auto_now=True,verbose_name='修改日期')

    def __str__(self):
        return self.caption
  
    class Meta:
        verbose_name = '元件套装表'
        verbose_name_plural = verbose_name

# 教程表
class TutorialList(models.Model):
    pid = models.AutoField(primary_key=True,verbose_name='教程_id')
    title = models.CharField(max_length=64,verbose_name='教程标题')
    description = models.CharField(null=True,blank=True, max_length=64,verbose_name='教程描述')
    componentSum = models.IntegerField(verbose_name='含有元件数',default=0)
    imgFile = models.FileField(null=True,blank=True,upload_to='tutotial/',verbose_name='教程img文件')
    key = models.CharField(null=True,blank=True,max_length=256,verbose_name='教程key')
    createTime_Internal = models.DateTimeField(auto_now_add=True,verbose_name='创建日期')
    alterTime_Internal = models.DateTimeField(auto_now=True,verbose_name='修改日期')

    def __str__(self):
        return self.title

    class Meta:
        verbose_name = '生成教程表'
        verbose_name_plural = verbose_name


