from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')

# 81 İl Listesi
iller = [
    "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Amasya", "Ankara", "Antalya", "Artvin", "Aydın", "Balıkesir", "Bilecik", "Bingöl", "Bitlis", "Bolu", "Burdur", "Bursa", "Çanakkale", "Çankırı", "Çorum", "Denizli", "Diyarbakır", "Edirne", "Elazığ", "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane", "Hakkari", "Hatay", "Isparta", "Mersin", "İstanbul", "İzmir", "Kars", "Kastamonu", "Kayseri", "Kırklareli", "Kırşehir", "Kocaeli", "Konya", "Kütahya", "Malatya", "Manisa", "Kahramanmaraş", "Mardin", "Muğla", "Muş", "Nevşehir", "Niğde", "Ordu", "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas", "Tekirdağ", "Tokat", "Trabzon", "Tunceli", "Şanlıurfa", "Uşak", "Van", "Yozgat", "Zonguldak", "Aksaray", "Bayburt", "Karaman", "Kırıkkale", "Batman", "Şırnak", "Bartın", "Ardahan", "Iğdır", "Yalova", "Karabük", "Kilis", "Osmaniye", "Düzce"
]

# Basit bir eğitim seti (Modelin çalışması için)
data = {
    'yas': [22, 25, 30, 35, 40, 45, 50, 55, 60, 33],
    'deneyim': [0, 3, 8, 13, 18, 23, 28, 33, 38, 10],
    'egitim': [1, 1, 2, 2, 1, 2, 1, 2, 2, 2],
    'maas': [20000, 30000, 45000, 55000, 65000, 75000, 80000, 90000, 100000, 55000]
}
df = pd.DataFrame(data)
X = df[['yas', 'deneyim', 'egitim']]
y = df['maas']
model = LinearRegression().fit(X, y)

@app.route('/')
def index():
    # İlleri alfabetik sıralayıp gönderiyoruz
    return render_template('index.html', iller=sorted(iller))

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    yas = int(input_data['yas'])
    deneyim = int(input_data['deneyim'])
    egitim = int(input_data['egitim'])
    secili_sehir = input_data['sehir']
    
    # Baz Tahmin
    baz_tahmin = model.predict([[yas, deneyim, egitim]])[0]
    
    # Şehir katsayıları simülasyonu (İstanbul en yüksek, diğerleri oranlı)
    grafik_verisi = {}
    for il in iller:
        # Şehre göre ufak çarpanlar (Örn: İstanbul %20 fazla, bazı iller %10 az gibi)
        carpan = 1.2 if il == "İstanbul" else (1.1 if il in ["Ankara", "İzmir", "Kocaeli"] else 0.9)
        tahmin = baz_tahmin * carpan
        grafik_verisi[il] = round(tahmin, 0)
    
    # Seçilen şehrin sonucunu döndür
    return jsonify({
        'tahmin': f"{int(grafik_verisi[secili_sehir]):,}".replace(",", "."),
        'grafik': grafik_verisi
    })

if __name__ == '__main__':
    app.run(debug=True)
