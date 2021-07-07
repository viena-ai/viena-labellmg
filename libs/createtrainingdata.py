import shutil,os,zipfile
import boto3
import requests
import json
import uuid
class createTraingdataSet:

    def __init__(self, projectFolderPath,imagesFolderPath,labelsFolderPath,classFilePath,noOfClasses,zipFilename):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        print("Inside create and upload class")
        self.status = ''

        self.ACCESS_KEY = 'AKIATJDN4DL7FS4WURNM'
        self.SECRET_KEY = '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'
        self.createZipdataset(imagesFolderPath,labelsFolderPath,classFilePath,noOfClasses)
        self.upload_to_aws(zipFilename, 'testbreezbucket', 'train_data.zip')

    def getStatus(self):
        return self.status

    def createZipdataset(self,imagesFolderPath,labelsFolderPath,classFilePath,noOfClasses):
        print("Inside class")
        if(os.path.isdir('train_data')):
            shutil.rmtree('train_data')
        if(os.path.isfile('train_data.zip')):
            os.remove('train_data.zip')
        os.mkdir('train_data')
        shutil.copytree(imagesFolderPath, 'train_data/images')
        shutil.copytree(labelsFolderPath, 'train_data/labels')
        shutil.copyfile(classFilePath, 'train_data/train_data.names')
        self.createDataFile(noOfClasses)
        self.createYoloConfig(noOfClasses)
        shutil.make_archive('train_data', 'zip', 'train_data')


    def createDataFile(self,noOfClasses):
        with open(os.path.abspath(os.getcwd()) + '/train_data/train.data', 'a') as the_file:
            the_file.write('classes= ' + str(noOfClasses) + '\n')
            the_file.write('train  = \n')
            the_file.write('valid  = \n')
            the_file.write('names  = train_data.names \n')
            the_file.write('backup  = \n')
            the_file.write('eval=coco' + '\n')

    def createYoloConfig(self, noOfClasses):
        filters = (int((noOfClasses) + 5) * 3)
        configFile = open(os.path.abspath(os.getcwd()) + '/yolov3.cfg', "r")
        list_of_lines = configFile.readlines()
        list_of_lines[609] = "classes=" + str(noOfClasses) + '\n'
        list_of_lines[695] = "classes=" + str(noOfClasses) + '\n'
        list_of_lines[782] = "classes=" + str(noOfClasses) + '\n'
        list_of_lines[602] = "filters=" + str(filters) + '\n'
        list_of_lines[688] = "filters=" + str(filters) + '\n'
        list_of_lines[775] = "filters=" + str(filters) + '\n'
        configFile = open('/yolov3.cfg', "w")
        configFile.writelines(list_of_lines)
        configFile.close()
        shutil.copyfile('/yolov3.cfg','train_data/yolov3.cfg')

    def upload_to_aws(self,zipFilename, bucket, s3_file):
        s3 = boto3.client('s3', aws_access_key_id=self.ACCESS_KEY,
                          aws_secret_access_key=self.SECRET_KEY)

        try:
            s3.upload_file(zipFilename, bucket, s3_file, ExtraArgs={'ACL':'public-read'})
            print("Upload Successful")
            self.status = 'True'
            return True
        except FileNotFoundError:
            self.status = 'false'
            print("The file was not found")
            return False

class postTrainingDetailsToRest:
    def __init__(self,zipFilename):
        self.statusResponse =''
        uniqueUuid = uuid.uuid1()
        print(uniqueUuid)
        data={
                "requestDetails": {
                "action": "START_TRAIN",
                "processId": str(uniqueUuid),
                "fileName": zipFilename,
                "archiveLink": "https://testbreezbucket.s3.amazonaws.com/train_data.zip"
                },
                "deviceCode": "ABCD_2"
                }
        data_json = json.dumps(data)
        headers = {'Content-type': 'application/json'}
        url = 'http://18.210.167.14:8080/vflow/upload_high_priority_req_res_msgs'
        try:
            response = requests.post(url, data=data_json, headers=headers)
            self.statusResponse = response
        except:
            self.statusResponse = response
            print("An exception occurred")


    def getPostStatus(self):
        return self.statusResponse


