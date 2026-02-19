# tensor

Bu proje, C dilinde düşük seviyeli bellek yönetimi kullanılarak geliştirilmiş, kısıtlı kaynaklara sahip sistemler için tasarlanmış bir yapay zeka çıkarım motorudur.

## 🚀 Özellikler
- **Mixed Precision:** FP32, FP16 ve INT8 veri tiplerini aynı işlemde kullanabilme.
- **Quantization:** Veriyi 8-bit tam sayılara sıkıştırarak bellek tasarrufu sağlama.
- **Memory Efficiency:** C `union` yapısı ile optimize edilmiş bellek alanı kullanımı.

## 🛠️ Kullanılan Araçlar
- **IDE:** Dev-C++ 5.11
- **Derleyici:** TDM-GCC 4.9.2
- **Dil Modeli:** Gemini 1.5 Flash (Algoritma optimizasyonu ve IEEE 754 dönüşümleri için).

## 🧠 Teknik Detaylar
- **Tensör Mimarisi:** Veriyi ve metadata bilgisini (scale, zero_point) tek bir struct altında toplar.
- **Dönüşüm Mantığı:** Float-to-Half ve Affine Quantization algoritmaları bit seviyesinde uygulanmıştır.

## 📝 Çalıştırma
`main.c` dosyasını GCC ile derleyip çalıştırabilirsiniz:
```bash
gcc main.c -o ai_inference
./ai_inference
