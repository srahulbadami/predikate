from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.forms.widgets import SelectDateWidget
import datetime
from .models import User
 
class SignUpForm(UserCreationForm):
	dob = forms.DateField(label='Date of birth',
									widget=SelectDateWidget(years=[y for y in range(1950,
																					datetime.datetime.now().year)],
															attrs={'class':"input--style-3",'style': 'width: 33%; display: inline-block;'}),)
	email = forms.EmailField(widget=forms.TextInput(
		attrs={
			'class':"input--style-3",'placeholder':"Email"
		}
	))
	first_name = forms.CharField(widget=forms.TextInput(
		attrs={
			'class':"input--style-3",'placeholder':"First Name"
		}
	))
	last_name = forms.CharField(widget=forms.TextInput(
		attrs={
			'class':"input--style-3",'placeholder':"Last Name"
		}
	))
	CHOICES= (
('M', 'Male'),
('F', 'Female'),
)
	gender = forms.CharField(label='Gender',widget=forms.Select(choices=CHOICES,
		attrs={
			'class':"input--style-3",
		}
		))
	password1=forms.CharField(widget=forms.PasswordInput(attrs={
			'class':"input--style-3",'placeholder':"Password"
		}))
	password2=forms.CharField(widget=forms.PasswordInput(attrs={
			'class':"input--style-3",'placeholder':"Confirm Password"
		}))
	class Meta:
		model = User
		fields = ('email','first_name','last_name','dob','gender')
