# بسته بازبینی‌شده مرحله اول پژوهش NHANES 2017–2018

هدف: دریافت و ادغام فایل‌های رسمی، انتخاب افراد واجد شرایط، کدگذاری متغیر هدف، کنترل کیفیت و تولید دو دیتاست اولیه با **حفظ مقادیر گمشده**. هیچ آموزش مدل، SHAP، یا جایگذاری مقادیر گمشده در این مرحله انجام نشده است.

## ساختار

- `raw_xpt/`: شش فایل XPT اصلی CDC؛ بدون تغییر و برای بازتولید نتایج.
- `prepare_nhanes.py`: کد اصلاح‌شده آماده‌سازی اولیه که هر دو دیتاست و جدول‌های اصلی را تولید می‌کند.
- `additional_audits.py`: کنترل‌های تکمیلی شامل علت نبودن فشار خون، وضعیت دور کمر، تعداد خواندن‌های فشار خون، و اصلاح خطای خواندن صفرهای XPT.
- `output/nhanes_stage1_clean_with_missing.csv`: جامعه ۵۳۹۳ نفریِ دارای پاسخ معتبر به DIQ010.
- `output/nhanes_examined_clean_with_missing.csv`: زیرگروه پیشنهادی ۵۰۹۸ نفریِ معاینه‌شده (هنوز انتخاب قطعی پژوهش نیست).
- `output/missingness_report.csv` و `output/examined_missingness_report.csv`: گزارش کامل داده‌های گمشده.
- `output/raw_file_sha256.txt`: هش فایل‌های XPT برای راستی‌آزمایی محتوای خام.
- `output/independent_qa_summary.csv`: خلاصه کنترل مستقل داده‌ها.
- `Gozarese_Momayeze_TaeidShode_Marhale1_NHANES.docx`: گزارش فارسی قابل بررسی با استاد شامل اعداد تأییدشده و تصمیم‌های نیازمند تأیید.

## اجرای مجدد در لینوکس / VS Code

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_nhanes.py
python additional_audits.py
```

اسکریپت‌ها از فایل‌های داخل `raw_xpt/` می‌خوانند و فایل‌های داخل `output/` را بازتولید می‌کنند. فایل‌های اصلی دستکاری نمی‌شوند.

## نکات مهم

- تنها پاسخ‌های `DIQ010 = 1` یا `2` به‌عنوان هدف وارد شده‌اند؛ وضعیت مرزی و سایر پاسخ‌های نامعتبر کنار گذاشته شده‌اند.
- در گروه معاینه‌شده ۵۰۹۸ فرد وجود دارد؛ ۴۰۳۱ نفر در ۱۱ پیش‌بین اولیه کامل و ۱۰۶۷ نفر دارای دست‌کم یک ویژگی گمشده هستند.
- `SEQN`، وزن‌های نمونه‌گیری و شناسه‌های طرح نمونه‌گیری پیش‌بین مدل نیستند.
- انتخاب جامعه ۵۰۹۸ نفری و فهرست ۱۱ ویژگی هنوز باید با استاد راهنما تأیید شود.
- روش فعلی فشار خون، میانگین قرائت‌های موجود از اول تا چهارم است؛ قاعده نهایی و حداقل تعداد خواندن‌های معتبر باید پیش از مدل‌سازی تثبیت شود.
- تفاوت مهم با نسخه اولیه: `pandas.read_sas` ممکن است صفر XPT را به مقدار `5.397605346934028e-79` تبدیل کند. نسخه اصلاح‌شده فقط همان مقدار دقیق را به صفر برمی‌گرداند. این اصلاح، تعداد افراد و آمار گمشدگی را تغییر نمی‌دهد.
- سن `RIDAGEYR=80` شامل همه افراد ۸۰ سال و بالاتر است. مقدار `INDFMPIR=5` نشان‌دهنده ۵ یا بیشتر است.
- تأیید عملکرد مدل، پایداری SHAP یا مکانیسم گمشدگی در این بسته انجام نشده است.

## منابع رسمی

- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.htm
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.htm
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.htm
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SMQ_J.htm
- https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PAQ_J.htm
- https://github.com/pandas-dev/pandas/issues/30051
