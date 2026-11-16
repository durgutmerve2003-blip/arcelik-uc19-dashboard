# UC19 · Parts prediction before the technician leaves

## Klasör yapısı
- `gen_data.py`  → dummy veriyi üretir (data/ klasörüne CSV yazar)
- `build.py`     → CSV'leri template.html içine gömer, out/ klasörüne dashboard'u yazar
- `template.html`→ dashboard'un kodu (HTML + CSS + JavaScript)
- `data/`        → veri seti (CSV'ler + data dictionary)
- `out/uc19_parts_prediction_dashboard.html` → bitmiş dashboard

## VS Code'da çalıştırma
1. Klasörü VS Code'da aç (File → Open Folder).
2. Terminal: `pip install pandas numpy`
3. Veriyi yeniden üretmek için: `python gen_data.py`
4. Dashboard'u yeniden oluşturmak için: `python build.py`
5. `out/uc19_parts_prediction_dashboard.html` dosyasını tarayıcıda aç
   (çift tıkla, ya da VS Code'da "Live Server" eklentisiyle sağ tık → Open with Live Server).

template.html'de bir şeyi değiştirince `python build.py` çalıştırıp sayfayı yenilemen yeterli.

## Canlı dashboard (GitHub Pages)
Kökteki `index.html`, out/ klasöründeki dashboard'un kopyasıdır; GitHub Pages bunu açar.
build.py'yi çalıştırdıktan sonra: `cp out/uc19_parts_prediction_dashboard.html index.html`
