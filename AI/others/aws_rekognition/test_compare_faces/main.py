from dotenv import load_dotenv
import os

import boto3


# Define constants
load_dotenv()
AWS_USERNAME = os.environ['aws_username']
AWS_BUCKET = os.environ['aws_bucket']
AWS_REGION = os.environ['aws_region']
AWS_ACCESS_KEY_ID = os.environ['aws_access_key_id']
AWS_SECRET_ACCESS_KEY = os.environ['aws_secret_access_key']

# Initialize rekognition client
aws_rekognition_client = boto3.client(
    service_name='rekognition',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
print('DBG: Client initialized')

# Preload images
img_folder_path = './img'

me_and_passport_img_filename = '2020-10-16-114752.jpg'
me_img_filename = '2020-10-16-114805.jpg'
passport_img_filename = '2020-10-16-114818.jpg'

me_and_passport_img_bytes = open('{0}/{1}'.format(img_folder_path, me_and_passport_img_filename), 'rb').read()
me_img_bytes = open('{0}/{1}'.format(img_folder_path, me_img_filename), 'rb').read()
passport_img_bytes = open('{0}/{1}'.format(img_folder_path, passport_img_filename), 'rb').read()

print('DBG: Images preloaded')

# Fetch compare_faces results
compare_faces_params = {
    'SourceImage': {
        'Bytes': me_img_bytes
    },
    'TargetImage': {
        'Bytes': passport_img_bytes
    },
    'SimilarityThreshold': 70,
    'QualityFilter': 'AUTO' # 'NONE'|'AUTO'|'LOW'|'MEDIUM'|'HIGH'
}

response = aws_rekognition_client.compare_faces(**compare_faces_params)
print(response)

print('DBG: Fetched response from compare_faces')

print('DBG: end of script')

