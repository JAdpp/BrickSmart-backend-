from django import forms

class PromptForm(forms.Form):
    prompt = forms.CharField(label="输入关键字", max_length=100)


class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='上传图片')