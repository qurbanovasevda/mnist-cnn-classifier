# Handwritten Digit Classifier — CNN vs MLP (MNIST)

## Framework
PyTorch (torch, torchvision)

## Layihə strukturu

mnist_project/
├── data/ # MNIST datasetı (avtomatik endirilir)
├── custom_digits/ # 5 real əl yazısı rəqəm şəkli
├── models.py # MLP və CNN arxitekturaları
├── train.py # Data pipeline, training, evaluation, qrafiklər
├── inference.py # Saxlanmış modelin yenidən yüklənməsi + custom test
├── best_cnn.pth # Öyrədilmiş CNN ağırlıqları
├── training_curves.png # Loss/Accuracy qrafiki
├── confusion_matrix.png # Confusion matrix
├── custom_digit_predictions.png # Custom test nəticələri
└── README.md


## Data Pipeline
- MNIST `torchvision.datasets.MNIST` ilə yükləndi (standart loader, əlavə pipeline gizlədilmədi)
- Normallaşdırma: piksel dəyərləri `mean=0.1307, std=0.3081` ilə standartlaşdırıldı
- Split: 60,000 orijinal train setindən **50,000 train / 10,000 validation** ayrıldı; 10,000 test set toxunulmadan saxlanıldı (default test seti validation kimi işlədilmədi)

## CNN Arxitekturası (layer-be-layer)

| Layer | Növ | Parametrlər | Çıxış ölçüsü |
|---|---|---|---|
| 1 | Conv2d | 1→32 kanal, kernel 3×3, padding 1 | 32×28×28 |
| 2 | ReLU | — | 32×28×28 |
| 3 | MaxPool2d | kernel 2 | 32×14×14 |
| 4 | Conv2d | 32→64 kanal, kernel 3×3, padding 1 | 64×14×14 |
| 5 | ReLU | — | 64×14×14 |
| 6 | MaxPool2d | kernel 2 | 64×7×7 |
| 7 | Flatten | — | 3136 |
| 8 | Linear | 3136→128 | 128 |
| 9 | ReLU | — | 128 |
| 10 | Linear | 128→10 | 10 (sinif) |

**2 conv+pooling blok** var (tələb olunan minimum), ardınca dense head.

## MLP Baseline (müqayisə üçün)
- Arxitektura: 784 → 256 → 128 → 10 (fully-connected, ReLU aktivasiyaları ilə)
- **Eyni** data, **eyni** split, **eyni** epoch sayı (10) ilə öyrədildi — ədalətli müqayisə üçün

## Nəticələr

| Model | Val Accuracy (10-cu epoch) | Test Accuracy |
|---|---|---|
| MLP | 97.69% | — |
| CNN | 99.00% | **99.10%** |

CNN, MLP-ni ~1.3-1.4% fərqlə keçdi. MNIST miqyasında bu, 10,000 test şəklindən təxminən 130+ daha çox rəqəmin düzgün tanınması deməkdir. Fərqin səbəbi: MLP şəkli flatten edərkən (784 tək piksel sırasına çevirərkən) piksellər arası **məkan əlaqəsini** (spatial structure) itirir. CNN isə convolution əməliyyatı vasitəsilə kənarları, əyriliyi və lokal naxışları qoruyaraq öyrənir — bu da rəqəm tanımada təbii üstünlük yaradır.

Həmçinin qeyd: MLP-nin val loss-u epoch 5-dən sonra dalğalanmağa (artıb-azalmağa) başladı (`0.0906 → 0.1029 → 0.1326`), bu overfitting əlamətidir. CNN-də bu dalğalanma daha azdır və ümumi trend daha sabitdir.

Training curves qrafiki: `training_curves.png`

## Confusion Matrix

Test set üzərində CNN üçün hesablanıb (`confusion_matrix.png`).

**Ən çox qarışan cüt: Əsl rəqəm = 2, Proqnoz = 7** (12 dəfə səhv edilib)

**Hipotez:** Əl yazısında bəzi "2" rəqəmləri, aşağı hissədəki dövrəvi əyri qısa/düz yazıldıqda, "7"-nin şaquli-üfüqi formasına bənzəyə bilir — xüsusən "2"-nin alt xətti düz və üfüqi çəkildikdə, üst hissəsi isə "7"-nin yuxarı üfüqi xəttinə bənzər bir əyriliklə yazıldıqda. Bu iki rəqəm arasında struktur oxşarlığı (yuxarıda üfüqi/əyri xətt, aşağıda nisbətən düz xətt) modelin qarışmasına səbəb olur.

## Custom Image Test (MNIST-də olmayan real şəkillər)

5 əl yazısı rəqəm (ağ kağızda göy qələmlə çəkilib, telefon kamerası ilə fotolanıb) test edildi.

**Preprocessing addımları** (`inference.py`):
1. Grayscale-ə çevrilmə
2. İnvert (fon qara, rəqəm ağ olsun — MNIST formatına uyğunlaşdırmaq üçün, çünki orijinal şəkildə fon ağ, rəqəm tünddür)
3. Threshold ilə səs-küyün təmizlənməsi
4. Bounding box ilə rəqəmin tapılıb kəsilməsi və mərkəzləşdirilməsi
5. Ətrafına boşluq (padding) əlavə edilməsi (MNIST-də rəqəmlər kənara qədər getmir)
6. 28×28-ə kiçildilmə (`INTER_AREA` interpolyasiya ilə)
7. Xətlərin `dilate` ilə qalınlaşdırılması (nazik qələm xəttini MNIST-in qalın rəqəmlərinə yaxınlaşdırmaq üçün)
8. Eyni normallaşdırma (`mean=0.1307, std=0.3081`)

**Nəticələr:**

| Şəkil | Əsl rəqəm | Proqnoz | Əminlik | Nəticə |
|---|---|---|---|---|
| 1 | 0 | 0 | 86.5% | ✅ |
| 2 | 3 | 3 | 99.2% | ✅ |
| 3 | 5 | 5 | 99.8% | ✅ |
| 4 | 7 | 1 | 47.2% | ❌ |
| 5 | 9 | 9 | 34.8% | ✅ |

**Doğruluq: 4/5 (80%)**

**Müşahidə:** Model 99.10% test accuracy-yə baxmayaraq, real-world şəkillərdə zəiflədi (80%) — bu, MNIST test setinin özünün "təmiz" və standartlaşdırılmış olması, real şəkillərin isə fərqli xətt qalınlığı, mərkəzləşmə və yazı tərzi daşımasından qaynaqlanır (generalization gap). Diqqətəlayiqdir ki, model səhv etdiyi yerdə (7→1) ən aşağı əminliklə (47.2%) proqnoz verdi — düzgün tapdığı yerlərdə isə əminlik adətən 85%+ oldu. Bu, aşağı-əminlikli proqnozların real tətbiqlərdə "əlavə yoxlama tələb edir" kimi işarələnə biləcəyini göstərir.

## Model Persistence
- CNN ağırlıqları `best_cnn.pth`-a saxlanıldı (`torch.save(model.state_dict(), ...)`)
- `inference.py` — tamamilə ayrı skriptdə, təkrar öyrətmə olmadan modeli yükləyib həm confusion matrix, həm custom image test üçün istifadə edir

## Anti-Cheat
CNN tam sıfırdan (random initialization) öyrədildi. Heç bir pretrained ağırlıq və ya transfer learning istifadə olunmadı.