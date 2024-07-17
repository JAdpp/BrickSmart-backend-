from django.contrib import admin

from .models import ModelTask, ComponentList, PackageList, TutorialList

# 注册ModelTask模型
admin.site.register(ModelTask)

# 注册ComponentList模型
admin.site.register(ComponentList)

# 注册PackageList模型
admin.site.register(PackageList)

# 注册TutorialList模型
admin.site.register(TutorialList)
