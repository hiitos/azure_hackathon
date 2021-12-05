import logging
import azure.functions as func
import sys
import time
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person
import io
import base64
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    #functionsから受け取るもの
    # {
    #     'user_id':str,
    #     'person_group':str,
    #     'images': base64のlist
    # }  

    #データの取得
    try:
        req_body = req.get_json()  # postが入る
    except ValueError:
        pass
    else:
        user_id = req_body['user_id']
        target_person_group = req_body['person_group']
        images = req_body['images']   #base64のlist

    # # 認証されたFaceClientを作成
    KEY = os.environ['Key']  #環境変数
    ENDPOINT = os.environ['Value']
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    # Person Groupの作成(PERSON GROUPがすでに登録されていないか確認しながら)
    person_group_list = face_client.person_group.list()
    count = 0
    for i in range(0, len(person_group_list)):
        if person_group_list[i].person_group_id == target_person_group:
            count += 1

    PERSON_GROUP_ID = target_person_group
    if count == 0:   #person_groupがすでに登録されていないか確認
        logging.info("{}というPerson_groupが登録されていないため新たに作成します".format(
            target_person_group))
        logging.info('_________________________')
        face_client.person_group.create(
            person_group_id=PERSON_GROUP_ID, name=PERSON_GROUP_ID)

    #Person オブジェクトの作成
    person = face_client.person_group_person.create(PERSON_GROUP_ID, user_id)
    if person:
        logging.info('Personオブジェクトの作成完了')
        logging.info('_________________________')

    #顔の登録
    for image in images:  # バイナリデータ
        image = image.split(',')[1]
        image = base64.b64decode(image)  # base64(bytes)
        image = io.BytesIO(image)

        faces = face_client.face.detect_with_stream(image)
        if len(faces) == 1:
            face_client.person_group_person.add_face_from_stream(
                PERSON_GROUP_ID, person.person_id, image)
            logging.info('顔の登録完了')
        else: 
            logging.info('写真から顔が検出されません')

    logging.info('_________________________')

    '''Train PersonGroup'''
    logging.info('Training the person group...')
    # Person groupを学習する
    face_client.person_group.train(PERSON_GROUP_ID)
    while (True):
        training_status = face_client.person_group.get_training_status(
            PERSON_GROUP_ID)
        logging.info("Training status: {}.".format(training_status.status))
        if (training_status.status is TrainingStatusType.succeeded):
            logging.info('Training Complete!!')
            break
        elif (training_status.status is TrainingStatusType.failed):
            face_client.person_group.delete(person_group_id=PERSON_GROUP_ID)
            sys.exit('Training the person group has failed.')
        time.sleep(5)
