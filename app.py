import cv2
from ultralytics import YOLO
from tinydb import TinyDB, Query

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64,uuid
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
db = TinyDB('db.json')

def read_plate(img):
    final = {"plato":None,"IR":None}
    try:
        # بارگذاری مدل YOLO برای شناسایی پلاک و مدل برای شناسایی کاراکترها
        ymodel = YOLO("best.pt")  # مدل YOLO برای شناسایی پلاک خودرو
        nmodel = YOLO("best_char2.pt")  # مدل YOLO برای شناسایی کاراکترهای پلاک

        # پردازش تصویر و شناسایی پلاک‌ها
        results = ymodel(img)  # شناسایی پلاک‌ها در تصویر
        res_plotted = results[0].plot()  # رسم نتایج بر روی تصویر (برای نمایش)

        # استخراج جعبه‌های محدودکننده برای پلاک‌ها
        boxes = results[0].boxes.xyxy.tolist()

        # جستجو در جعبه‌ها و برش تصویر پلاک
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            plate_img = img[int(y1):int(y2), int(x1):int(x2)]  # برش پلاک از تصویر

        # شناسایی کاراکترها در پلاک
        char_results = nmodel(plate_img)  # شناسایی کاراکترها با مدل YOLO مخصوص کاراکتر
        if len(char_results[0].boxes) > 0:
            detected_characters = []
            char_bboxes = []  # لیستی برای ذخیره جعبه‌های محدودکننده کاراکترها
            for c_box in char_results[0].boxes:
                label = char_results[0].names[int(c_box.cls)]  # برچسب کاراکتر
                x_coord = c_box.xyxy[0][0]  # مختصات X برای مرتب‌سازی
                detected_characters.append((x_coord, label))  # ذخیره کاراکتر و مختصات آن
                char_bbox = list(map(int, c_box.xyxy[0]))  # ذخیره جعبه محدودکننده کاراکتر
                char_bboxes.append(char_bbox)

            # مرتب‌سازی کاراکترها بر اساس مختصات X (برای ساخت پلاک کامل)
            detected_characters.sort(key=lambda x: x[0])
            # استخراج متن پلاک از کاراکترهای شناسایی‌شده
            number = ""
            i = 0
            
            for char in detected_characters:
                if i == 6:  # افزودن "IRAN:" بعد از شش کاراکتر اول
                    final["plato"] = number
                    number += " IRAN:"
                number += "".join(char[1])  # اضافه کردن کاراکتر به متن پلاک
                i += 1
        final["IR"] = number[-2:]
        return final  # بازگشت پلاک شناسایی‌شده
    except Exception as e:
        # در صورت بروز خطا، پیغام خطا برمی‌گرداند
        return f"error => {str(e)}"

def character_parser(En,Pr):
    Chars = [{"ب":"B"},{"پ":"P"},{"ت":"T"},{"ث":"SS"},{"ج":"J"},{"د":"D"},{"ز":"Z"},{"س":"S"},{"ش":"SH"},{"ص":"SSS"},{"ط":"T"},{"ع":"EIN"},{"ف":"F"},{"ق":"GH"},{"ک":"K"},{"گ":"G"},{"ل":"L"},{"م":"M"},{"ن":"N"},{"و":"V"},{"ه":"H"},{"ی":"Y"}]
    if En:
        for dic in Chars:
            for key in dic:
                print(key)
                En = En.replace(dic[key],key)
        return En
     
    elif Pr:
        for dic in Chars:
            for key in dic:
                Pr = Pr.replace(key,dic[key])
        return Pr


# Directory to save uploaded images
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# First endpoint: Receives a base64 image
@app.route('/upload_image', methods=['POST'])
@cross_origin(supports_credentials=True)
def upload_image():
    data = request.get_json()
    if 'image_base64' not in data:
        return jsonify({'error': 'No image_base64 field in request'}), 400

    try:
        image_data = base64.urlsafe_b64decode(data['image_base64'])
        image_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.png')
        with open(image_path, 'wb') as image_file:
            image_file.write(image_data)

        image = cv2.imread(image_path)
        plate_number = read_plate(image)
        platos = Query()
        res = db.search((platos.plato == plate_number["plato"]) & (platos.IR == plate_number["IR"]))
        if res:
            return jsonify({'exist': True, 'id': res[0]["id"]}), 200
        
        return jsonify({'exist': False}), 400
    
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

@app.route('/add_plato', methods=['POST'])
@cross_origin(supports_credentials=True)
def add():
    data = request.get_json()
    if not all(key in data for key in ('plato', 'IR', 'name')):
        return jsonify({'error': 'Request must contain plato and IR and name'}), 400

    plato = data['plato']
    plato = character_parser(Pr=plato, En=None)
    
    IR = data['IR']
    name = data['name']

    id = str(uuid.uuid4())
    db.insert({'id':id, 'name': name, 'plato': plato, "IR": IR})

    return jsonify({'status': 'success'}), 200

@app.route('/remove_plato', methods=['POST'])
@cross_origin(supports_credentials=True)
def remove():
    data = request.get_json()
    print(data["id"])
    if not data["id"]:
        return jsonify({'error': 'Request must contain id'}), 400

    id = data['id']

    platos = Query()
    if db.remove(platos.id == id):
        return jsonify({'status': 'success'}), 200
    else:
        return jsonify({'status': 'not found'}), 400


@app.route('/plato_list', methods=['GET'])
@cross_origin(supports_credentials=True)
def process_param():
    all = db.all()
    print(character_parser(En="AAA",Pr=None))
    for i in range(len(all)): 
        print(type(all[i]["plato"]))
        all[i]["plato"] = character_parser(Pr=None,En=all[i]["plato"])

    return jsonify(all), 200

if __name__ == '__main__':
    app.run(debug=True)