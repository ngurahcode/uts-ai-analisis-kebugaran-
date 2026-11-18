import json
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#TAHAP 1: LINEAR REGRESSION (Prediksi Kalori)
X_latih_kalori = [
    [60, 165, 20], [70, 170, 30], [80, 175, 45],
    [55, 160, 15], [90, 180, 60], [75, 172, 40],
]
y_latih_kalori = [1200, 2100, 3200, 900, 4500, 2800]

model_regresi = LinearRegression()
model_regresi.fit(X_latih_kalori, y_latih_kalori)


#TAHAP 2: KNN (Fitur: [Skor BMI, Kalori Mingguan, Kode Tujuan])
# Kode Tujuan: 1 = Turunin Lemak, 2 = Nambah Massa Otot, 3 = Jaga Kebugaran / Pemulihan
X_latih_program = [
    # Fokus Bakar Lemak
    [28.5, 3500, 1], [27.0, 2800, 1], [31.0, 3200, 1], [24.5, 2900, 1],
    # Pembentukan Otot
    [22.0, 2200, 2], [20.5, 2000, 2], [23.5, 2500, 2], [18.0, 1800, 2],
    # Pemulihan & Surplus Nutrisi (Kurus ekstrem / kalori terlalu tinggi / pemulihan)
    [16.0, 3300, 1], [15.5, 2500, 2], [22.0, 4400, 3], [21.5, 1500, 3],
]

y_latih_program = [
    "Fokus Bakar Lemak", "Fokus Bakar Lemak", "Fokus Bakar Lemak", "Fokus Bakar Lemak",
    "Pembentukan Otot", "Pembentukan Otot", "Pembentukan Otot", "Pembentukan Otot",
    "Pemulihan & Surplus Nutrisi", "Pemulihan & Surplus Nutrisi", "Pemulihan & Surplus Nutrisi", "Pemulihan & Surplus Nutrisi",
]

scaler = StandardScaler()
X_latih_scaled = scaler.fit_transform(X_latih_program)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_latih_scaled, y_latih_program)

DETAIL_PROGRAM = {
    "Fokus Bakar Lemak": {
        "warna": "badge-red",
        "saran": "Sesuai targetmu untuk memangkas lemak, pertahankan defisit kalori ringan 300–500 kkal/hari dan kombinasikan lari rutin dengan latihan beban 3x seminggu."
    },
    "Pembentukan Otot": {
        "warna": "badge-blue",
        "saran": "Sesuai target massa otot, fokus latihan beban (hypertrophy) 3–4x seminggu, kurangi kardio berlebihan, dan penuhi protein 1,6–2g per kg berat badan."
    },
    "Pemulihan & Surplus Nutrisi": {
        "warna": "badge-green",
        "saran": "AI mendeteksi BMI terlalu rendah atau beban lari terlalu ekstrem dibanding targetmu. Prioritaskan istirahat cukup, kurangi lari berat, dan tingkatkan asupan kalori harian."
    }
}


#RUTE HALAMAN WEB
@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    # Siapkan titik data latih untuk ditampilkan di grafik
    data_grafik = []
    for fitur, label in zip(X_latih_program, y_latih_program):
        data_grafik.append({"x": fitur[0], "y": fitur[1], "label": label})

    if request.method == "POST":
        berat = float(request.form["berat"])
        tinggi = float(request.form["tinggi"])
        durasi = float(request.form["durasi"])
        tujuan = int(request.form["tujuan"])

        # Hitung BMI otomatis
        tinggi_meter = tinggi / 100
        bmi = round(berat / (tinggi_meter ** 2), 1)
        if bmi < 18.5:
            kategori_bmi = "Kurus (Underweight)"
        elif bmi < 25:
            kategori_bmi = "Ideal (Normal)"
        elif bmi < 30:
            kategori_bmi = "Berlebih (Overweight)"
        else:
            kategori_bmi = "Obesitas"

        # Tahap 1: Prediksi Kalori
        prediksi_kalori = max(0, model_regresi.predict([[berat, tinggi, durasi]])[0])

        # Tahap 2: Normalisasi [BMI, Kalori, Tujuan] -> Prediksi KNN
        fitur_baru_scaled = scaler.transform([[bmi, prediksi_kalori, tujuan]])
        rekomendasi = model_knn.predict(fitur_baru_scaled)[0]
        info_program = DETAIL_PROGRAM[rekomendasi]

        hasil = {
            "berat": int(berat) if berat.is_integer() else berat,
            "tinggi": int(tinggi) if tinggi.is_integer() else tinggi,
            "durasi": int(durasi) if durasi.is_integer() else durasi,
            "tujuan": tujuan,
            "bmi": bmi,
            "kategori_bmi": kategori_bmi,
            "kalori": round(prediksi_kalori),
            "program": rekomendasi,
            "warna": info_program["warna"],
            "saran": info_program["saran"]
        }

    return render_template("index.html", hasil=hasil, data_grafik=json.dumps(data_grafik))

if __name__ == "__main__":
    app.run(debug=True)