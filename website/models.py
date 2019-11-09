from django.utils import timezone
from django.db import models
from django.contrib.auth.models import (
    BaseUserManager, AbstractBaseUser
)
from .validators import validate_file_extension
from django.core.validators import FileExtensionValidator
class UserManager(BaseUserManager):
    def create_user(self, email, last_name, first_name, password=None, is_active=True, is_staff=False, is_admin=False):
        """
        Creates and saves a User with the given email and password.
        """
        if not email:
            raise ValueError('Users must have an email address')

        if not password:
            raise ValueError('Users must have a password')

        user_obj = self.model(
            email=self.normalize_email(email),
        )

        user_obj.set_password(password) # change user password
        user_obj.first_name = first_name
        user_obj.last_name = last_name
        user_obj.staff = is_staff
        user_obj.admin = is_admin
        user_obj.active = is_active
        user_obj.save()
        return user_obj

    def create_staffuser(self, email, password):
        """
        Creates and saves a staff user with the given email and password.
        """
        user = self.create_user(
            email=email,
            password=password,
        )
        user.staff = True
        user.save()
        return user

    def create_superuser(self, email,last_name, first_name, password):
        """
        Creates and saves a superuser with the given email and password.
        """
        user = self.create_user(
            email=email,
            last_name=last_name,
            first_name=first_name,
            password=password,
        )
        user.staff = True
        user.admin = True
        user.save()
        return user


class User(AbstractBaseUser):
    email = models.EmailField(
        verbose_name='email address',
        max_length=255,
        unique=True,
    )
    first_name = models.CharField(max_length=255, blank=False, null=False)
    last_name = models.CharField(max_length=255, blank=False, null=False)
    dob = models.DateField(max_length=8,null=True)
    gender = models.CharField(max_length=1,null=True)
    premiumUser = models.BooleanField(default=False)
    active = models.BooleanField(default=True)
    staff = models.BooleanField(default=False) # a admin user; non super-user
    admin = models.BooleanField(default=False) # a superuser
    timestamp = models.DateTimeField(auto_now_add=True)
    confirmedEmail = models.BooleanField(default=False) # Check if user is valid
    confirmedDate = models.DateTimeField(null=True)  # Check if user is valid
    # notice the absence of a "Password field", that's built in.

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']  # Email & Password are required by default.

    def get_full_name(self):
        # The user is identified by their email address
        return "%s %s" % (self.first_name, self.last_name)

    def get_short_name(self):
        # The user is identified by their email address
        return self.email

    def __str__(self):              # __unicode__ on Python 2
        return self.email

    def has_perm(self, perm, obj=None):
        "Does the user have a specific permission?"
        # Simplest possible answer: Yes, always
        return True

    def has_module_perms(self, app_label):
        "Does the user have permissions to view the app `app_label`?"
        # Simplest possible answer: Yes, always
        return True

    @property
    def is_staff(self):
        "Is the user a member of staff?"
        return self.staff

    @property
    def is_admin(self):
        "Is the user a admin member?"
        return self.admin

    @property
    def is_active(self):
        "Is the user active?"
        return self.active

    objects = UserManager()



class CustomModels(models.Model):
    name = models.CharField(max_length=255, blank=False, null=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    upload = models.FileField(upload_to ='datasets/%Y/%m/%d/', validators=[FileExtensionValidator(allowed_extensions=['csv','tsv'])]) 
    model = models.CharField(max_length=255, blank=True, null=True,)
    timestamp = models.DateTimeField(auto_now_add=True)
    custom_url = models.CharField(max_length=255, blank=False, null=False)