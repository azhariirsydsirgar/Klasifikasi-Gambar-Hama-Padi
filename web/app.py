from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

dic = {0: 'Hama Putih Palsu',  
       1: 'Penggerek Batang Padi Bergaris',
       2: 'Penggerek Batang Padi Kuning', 
       3: 'Walangsangit',
       4: 'Wereng Coklat',
       5: 'Wereng Hijau'}

model = load_model('Azhari-Hama Exception.h5') 
model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 224, 224, 3)
    
    # Mendapatkan probabilitas untuk semua kelas
    probs = model.predict(i)
    # Mengambil indeks kelas dengan probabilitas tertinggi
    predicted_class = probs.argmax()
    
    # Mengembalikan nama kelas yang sesuai dengan indeks
    return dic[predicted_class]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        
    return render_template("classification.html", prediction=p, img_path=img_path)

if __name__ =='__main__':
    app.run(debug=True)
