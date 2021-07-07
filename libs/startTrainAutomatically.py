import shutil,os,zipfile
from pathlib import Path

import boto3
import requests
import json
import uuid
import cv2
from labelImg import MainWindow
from PyQt5.QtWidgets import QMessageBox


class startTrainAutomatically:
    def __init__(self,checkfortrain,parent=MainWindow):
        self.ACCESS_KEY = 'AKIATJDN4DL7FS4WURNM'
        self.SECRET_KEY = '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'
        self.status = ''
        self.statusResponse = ''
        if checkfortrain >= 2:
            print("creating data")
            logstatus="creating data"
            color='#000000'
            method = MainWindow.autoUploadandandAnnotateLogs(self,logstatus,color)

            file1 = open('annotatedData.txt', 'r')
            count = 0
            if not os.path.exists('train_data'):
                os.mkdir('train_data')
            else:
                shutil.rmtree("train_data")
                os.mkdir('train_data')

            dest_path = os.path.abspath(os.getcwd())
            if not os.path.exists(os.path.join(dest_path + "/train_data", "images")):
                os.makedirs(os.path.join(dest_path + "/train_data", "images"))
                os.makedirs(os.path.join(dest_path + "/train_data", "labels"))

            while True:
                count += 1
                line = file1.readline()
                if not line:
                    break
                path = line.strip()
                if (path.endswith('.txt')):

                    filename = Path(path).name
                    filename_wo_ext, file_extension = os.path.splitext(filename)
                    j = 0
                    for j in range(20):
                        labeltarget = r'' + dest_path + '/train_data/labels/' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        labelaveraging_target = r'' + dest_path + '/train_data/labels/averaging_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        labelgausian_target = r'' + dest_path + '/train_data/labels/gausian_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        labelbilateral_target = r'' + dest_path + '/train_data/labels/bilateral_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        labelmedian_target = r'' + dest_path + '/train_data/labels/median_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        target = r'' + dest_path + '/train_data/images/' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        averaging_target = r'' + dest_path + '/train_data/images/averaging_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        gausian_target = r'' + dest_path + '/train_data/images/gausian_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        bilateral_target = r'' + dest_path + '/train_data/images/bilateral_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        median_target = r'' + dest_path + '/train_data/images/median_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        shutil.copyfile(path, target)
                        shutil.copyfile(path, averaging_target)
                        shutil.copyfile(path, gausian_target)
                        shutil.copyfile(path, bilateral_target)
                        shutil.copyfile(path, median_target)
                        shutil.copyfile(path, labeltarget)
                        shutil.copyfile(path, labelaveraging_target)
                        shutil.copyfile(path, labelgausian_target)
                        shutil.copyfile(path, labelbilateral_target)
                        shutil.copyfile(path, labelmedian_target)
                        j += 1
                else:
                    try:
                        #print(path)
                        img = cv2.imread(path)
                        averaging = cv2.blur(img, (5, 5))
                        gaussian = cv2.GaussianBlur(img, (5, 5), 1)
                        median = cv2.medianBlur(img, 5)
                        bilateral = cv2.bilateralFilter(img, 9, 75, 75)
                        filename = Path(path).name
                        filename_wo_ext, file_extension = os.path.splitext(filename)
                        dest_path = os.path.abspath(os.getcwd())
                        if not os.path.exists(os.path.join(dest_path + "/train_data", "images")):
                            os.makedirs(os.path.join(dest_path + "/train_data", "images"))
                        j = 0
                        for j in range(20):
                            cv2.imwrite(dest_path + '/train_data/images/' + filename_wo_ext + "_" + str(
                                j) + file_extension, img)
                            cv2.imwrite(dest_path + '/train_data/images/averaging_' + filename_wo_ext + "_" + str(
                                j) + file_extension, averaging)
                            cv2.imwrite(dest_path + '/train_data/images/gausian_' + filename_wo_ext + "_" + str(
                                j) + file_extension, gaussian)
                            cv2.imwrite(dest_path + '/train_data/images/bilateral_' + filename_wo_ext + "_" + str(
                                j) + file_extension, bilateral)
                            cv2.imwrite(dest_path + '/train_data/images/median_' + filename_wo_ext + "_" + str(
                                j) + file_extension, median)
                            j += 1
                    except:
                        print("This is an error message!")

            file1.close()
            # create .names file
            class_path = os.path.abspath(os.getcwd()) + '/data/predefined_classes.txt'
            #print("classpath")
            #print(class_path)
            names_path = os.path.abspath(os.getcwd()) + '/train_data/train_data.names'
            shutil.copyfile(class_path, names_path)

            # create .data file
            self.createDataFile()

            # make change in the config file
            self.changeYoloCfgFile()
            zipFilename="train_data.zip"
            shutil.make_archive('train_data', 'zip', 'train_data')
            print("successfully created the data")
            self.upload_to_aws(zipFilename, 'testbreezbucket', 'train_data.zip')
            if(self.status == 'True'):
                self.postTrainingDetailsToRestwithopencvdata(zipFilename)

    def getStatus(self):
        return self.status

    def createDataFile(self):
        num_lines = sum(1 for line in open(os.path.abspath(os.getcwd()) + '/data/predefined_classes.txt'))
        with open(os.path.abspath(os.getcwd()) + '/train_data/train.data', 'a') as the_file:
            the_file.write('classes= ' + str(num_lines) + '\n')
            the_file.write('train  = ' + str(os.path.abspath(os.getcwd()) + '/train_data/train.txt') + '\n')
            the_file.write('valid  = ' + str(os.path.abspath(os.getcwd()) + '/train_data/valid.txt') + '\n')
            the_file.write('names  = ' + str(os.path.abspath(os.getcwd()) + '/train_data/training.names') + '\n')
            the_file.write('backup  = ' + str(os.path.abspath(os.getcwd()) + '/train_data/backup/') + '\n')
            the_file.write('eval=coco' + '\n')

    def changeYoloCfgFile(self):
        numOfClasses = sum(1 for line in open(os.path.abspath(os.getcwd()) + '/data/predefined_classes.txt'))
        filters = ((numOfClasses + 5) * 3)

        a_file = open(os.path.abspath(os.getcwd()) + '/yolov3.cfg', "r")
        list_of_lines = a_file.readlines()
        list_of_lines[609] = "classes=" + str(numOfClasses) + '\n'
        list_of_lines[695] = "classes=" + str(numOfClasses) + '\n'
        list_of_lines[782] = "classes=" + str(numOfClasses) + '\n'
        list_of_lines[602] = "filters=" + str(filters) + '\n'
        list_of_lines[688] = "filters=" + str(filters) + '\n'
        list_of_lines[775] = "filters=" + str(filters) + '\n'

        a_file = open(os.path.abspath(os.getcwd()) + '/yolov3.cfg', "w")
        a_file.writelines(list_of_lines)
        a_file.close()
        shutil.copyfile(os.path.abspath(os.getcwd()) + '/yolov3.cfg', os.path.abspath(os.getcwd()) + '/train_data/yolov3.cfg')

    def upload_to_aws(self,zipFilename, bucket, s3_file):
        print("Uploading to S3")
        s3 = boto3.client('s3', aws_access_key_id=self.ACCESS_KEY,
                          aws_secret_access_key=self.SECRET_KEY)

        try:
            s3.upload_file(zipFilename, bucket, s3_file, ExtraArgs={'ACL':'public-read'})
            print("Upload Successful to S3")
            self.status = 'True'
            return True
        except FileNotFoundError:
            self.status = 'false'
            print("The file was not found")
            return False


    def postTrainingDetailsToRestwithopencvdata(self,zipFilename):
        uniqueUuid = uuid.uuid1()
        data={
                "requestDetails": {
                "action": "START_TRAIN",
                "uuid": str(uniqueUuid),
                "fileName": zipFilename,
                "archiveLink": "https://testbreezbucket.s3.amazonaws.com/train_data.zip"
                },
                "deviceCode": "ABCD"
                }
        data_json = json.dumps(data)
        headers = {'Content-type': 'application/json'}
        url = 'http://18.210.167.14:8080/vflow/upload_high_priority_req_res_msgs'
        try:
            response = requests.post(url, data=data_json, headers=headers)
            self.status = "Successfully posted to endpoint"
        except:
            self.status = response
            print("An exception occurred")



