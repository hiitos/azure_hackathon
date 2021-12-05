import logging
import azure.functions as func
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import io
import os
#from flask import jsonify
import json
import base64

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    #ラズパイから画像が一枚と探すPersonGroup
    try:
        req_body = req.get_json()  # postが入る
    except ValueError:
        pass
    else:
        image = req_body['image']  #str
        target_person_group = req_body['person_group']  # 写真を撮った場所でいい(placeでもいい)

    # 認証されたFaceClientを作成
    KEY = os.environ['Key']  # functions内で値を入れておく
    ENDPOINT = os.environ['Value']
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
    # Detect faces (きた画像に顔が写っているかどうかを確認)
    face_ids = {}  # faceIDとemotionを格納
    params = ["age", "gender", "smile", "glasses", "emotion"]
    
    image = base64.b64decode(image)# base64(bytes)
    image = io.BytesIO(image)  #_io.BytesIO

    faces = face_client.face.detect_with_stream(
        image,  return_face_attributes=params)
    if not faces:
        logging.info('Error:facesが読み込めていない')  # コメントアウトしてる
        return {}

    for face in faces:
        face_ids[face.face_id] = face.face_attributes.emotion.as_dict()

    if len(face_ids) == 0:
        logging.info('写真から顔の検出がない')
        return {}

    #Identify Face
    PERSON_GROUP_ID = target_person_group
    results = face_client.face.identify(
        list(face_ids.keys()), PERSON_GROUP_ID)  # デフォルトで'social-residence'にしてもらう
    if not results:
        logging.info('PersonGroupの顔が特定できない.')
        return {}
    else:
        logging.info('PersonGroupの顔が特定成功.')
    
    total_data = {}  # faceidとpersonIDを格納(結びつける)
    for person in results:
        if len(person.candidates) > 0:
            logging.info('face ID {} は、この画像に信頼度{}で含まれている.'
                  .format(person.face_id,person.candidates[0].confidence))

            # PERSON IDが一致するuser_idを取り出す
            person_group_list = face_client.person_group_person.list(
                PERSON_GROUP_ID)
            for i in range(0, len(person_group_list)):
                if person_group_list[i].person_id == person.candidates[0].person_id:
                    #person_group_list[i].persisted_face_idsのリストに追加する?　追加するならconfidenceの基準を設け追加した方がいい
                    total_data[person.face_id] = person_group_list[i].name
                    logging.info('personID:{}と顔が一致.'.format(
                        person.candidates[0].person_id))
                    logging.info('user_id：{}'.format(
                        person_group_list[i].name))
                    logging.info('-----------------------------')
        else:
            logging.info('この画像にfeceID{}を特定できる人がいない.'.format(
                person.face_id))

    if len(total_data) == 0:
        logging.info('検出された顔が誰なのか特定できなかった')
        return {}

    #user_idとemotionを結びつける
    submit_data = {}
    for key, value in total_data.items():
        face_id_get = key
        user_id = value
        emotion = face_ids.get(face_id_get)  # 感情値
        submit_data[user_id] = emotion

    logging.info(submit_data)
    logging.info('******************************')

    if submit_data:
        logging.info('Success!!!')
    else:
        logging.info('Failed')
    return func.HttpResponse(json.dumps(submit_data))
