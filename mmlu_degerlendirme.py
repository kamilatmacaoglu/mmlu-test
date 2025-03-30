import json
from transformers import pipeline

def mmlu_degerlendir(model_adi, veri_dosyasi, sonuc_dosyasi):
    """
    Belirtilen modeli MMLU veri seti üzerinde değerlendirir ve sonuçları kaydeder.

    Args:
        model_adi (str): Kullanılacak Hugging Face modelinin adı.
        veri_dosyasi (str): MMLU veri setinin bulunduğu JSON dosyasının yolu.
        sonuc_dosyasi (str): Sonuçların kaydedileceği JSON dosyasının yolu.
    """

    nlp = pipeline("text-classification", model=model_adi)

    with open(veri_dosyasi, "r") as f:
        veri = json.load(f)

    sonuclar = []
    dogru_sayisi = 0

    for soru_veri in veri:
        soru = soru_veri["soru"]
        cevaplar = soru_veri["cevaplar"]
        dogru_cevap = soru_veri["dogru_cevap"]

        model_cevaplari = []
        for cevap in cevaplar:
            sonuc = nlp(soru + " " + cevap)[0]
            model_cevaplari.append((cevap, sonuc["score"]))

        model_cevaplari.sort(key=lambda x: x[1], reverse=True)
        modelin_tahmini = model_cevaplari[0][0]

        if modelin_tahmini == dogru_cevap:
            dogru_sayisi += 1

        sonuclar.append({
            "soru": soru,
            "cevaplar": cevaplar,
            "dogru_cevap": dogru_cevap,
            "modelin_tahmini": modelin_tahmini
        })

    dogruluk = dogru_sayisi / len(veri)

    sonuc_verisi = {
        "model_adi": model_adi,
        "toplam_soru": len(veri),
        "dogru_sayisi": dogru_sayisi,
        "dogruluk": dogruluk,
        "sonuclar": sonuclar
    }

    with open(sonuc_dosyasi, "w") as f:
        json.dump(sonuc_verisi, f, indent=4)

    print(f"Model: {model_adi}")
    print(f"Doğruluk: {dogruluk}")

# Örnek kullanım
mmlu_degerlendir(
    model_adi="distilbert-base-uncased-finetuned-sst-2-english",
    veri_dosyasi="veri/mmlu_veri_seti.json",
    sonuc_dosyasi="sonuclar/model_sonuclari.json"
)