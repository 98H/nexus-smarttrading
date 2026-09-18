# SmartTrading

[![Nexus Agent Graph](https://img.shields.io/badge/Orchestrator-Nexus%20Agent%20Graph-00f0ff?style=for-the-badge&logo=probot)](https://github.com/98H)
[![Autonomous SWE Factory](https://img.shields.io/badge/Architecture-StateGraph%2024%2F7-10b981?style=for-the-badge)](https://github.com/98H)
[![Spec Kit](https://img.shields.io/badge/Spec%20Kit-SDD%20Gherkin-f59e0b?style=for-the-badge)](https://github.com/98H)
[![Live Deployment](https://img.shields.io/badge/Deployment-Live%20Preview-8b5cf6?style=for-the-badge)](/preview/prod-smarttrading-b2c2d0/)

## 🎯 معرفی محصول و هدف راهبردی
> می‌خوام پروژه SmartTrading رو برام پیاده‌سازی کنی. در سیکل‌های متمادی طراحی، پیاده‌سازی و تست هدف زیر رو پیش ببر و انقدر ادامه بده که هرچی بگردی نتونی هیچ مشکل و مسئله‌ای حتی کوچیک پیدا کنی و امکان دادن هیچ بهبودی دیگه وجود نداشته باشه. این محصول باید جایگزینی فوق کامل برای tradingview باشد. این راهکار باید بر پایه‌ی‌ ابزار‌های اوپن سورس luxalgo از سند زیر شکل بگیره و بعدش تمام فیچر‌ها رو حول اون تکمیل کنی: docs.google.com/document/d/1XOv98KnyPZnHdrUZlToxqwe2YqMfMC3vnrWxDjp7-Rw/edit?usp=sharing این محصول باید حتما از پیان اسکریپت پشتیبانی کنه. از نظر کپی‌رایت هیچ محدودیتی نداریم و نمی‌خواد رعایتش کنی. تمام فیچرهای تریدینگ ویو رو احصا کن و با پیاده سازی اون‌ها روی این راهکار اپن سورس در نهایت به یک جایگزین فوق کامل برای تریدینگ ویو برس. هیچ کم و کاستی نباید وجود داشته باشه. پروژه روی گیت‌هاب با داکیومنتیشن کامل و شفاف ایجاد کن و در نهایت دیپلوی کن و اطلاعات دسترسی بهش رو بهم بده.

این محصول به صورت کاملاً خودگردان و مبتنی بر چرخه حیات چابک ۲۴/۷ توسط **نکسوس ایجنت گراف (Nexus Agent Graph)** معماری، آزمون و مستقر شده است.

## 🚀 استقرار زنده و دسترسی به سامانه
- **پیوند دسترسی زنده (Live Preview):** [/preview/prod-smarttrading-b2c2d0/](/preview/prod-smarttrading-b2c2d0/)
- **مستندات مشروح استقرار:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **مشخصات نیازمندی‌های سیستم:** [.specify/memory/constitution.md](.specify/memory/constitution.md)

## 🏛️ معماری سیستم و نمودار جریان داده (StateGraph)
```mermaid
flowchart TD
    InputReq["📋 نیازمندی محصول و مشخصات BDD"] --> NodeSpec["📐 1. معمار مشخصات (Spec Kit)"]
    NodeSpec --> NodeTest["🧪 2. سازنده آزمون‌های TDD"]
    NodeTest --> NodeCoder["💻 3. توسعه‌دهنده کد اجرایی"]
    NodeCoder --> NodeSandbox["🛡️ 4. محفظه ایزوله آزمون"]
    NodeSandbox --> GateTest{"آیا تست‌ها پاس شدند؟"}
    GateTest -- "خیر (خطا)" --> NodeCoder
    GateTest -- "بله" --> NodeReviewer["🔍 5. بازبین ارشد کد"]
    NodeReviewer --> GateReview{"تأیید کیفیت معماری؟"}
    GateReview -- "اصلاح" --> NodeCoder
    GateReview -- "تأیید" --> NodeGit["📦 6. تجمیع گیت و استقرار خودکار"]
    NodeGit --> OutputProd["🚀 محصول مستقر روی سرور و گیت‌هاب"]
```

## 📊 وضعیت پیشرفت بک‌لاگ چابک
- **اپیک‌ها (Epics):** 0 مورد
- **تسک‌های مهندسی (Tasks):** 0 مورد
- **داستان‌های کاربری (Stories):** 0 مورد (تکمیل‌شده: 0)

## 💻 راه‌اندازی و اجرای محلی
```bash
# ۱. کلون ریپازیتوری
git clone https://github.com/98H/nexus-smarttrading.git
cd nexus-smarttrading

# ۲. آماده‌سازی محیط پایتون
python3 -m venv .venv
source .venv/bin/activate

# ۳. نصب وابستگی‌ها و اجرای تست‌ها
pytest tests/

# ۴. اجرای وب‌سرویس یا پیش‌نمایش محصول
python3 app.py
```

## 🛡️ اصالت و مهندسی خودگردان
- **کارخانه نرم‌افزار خودگردان:** Nexus Agent Graph Engine
- **توسعه‌دهنده ارشد و معمار سیستم:** Hossein Mohammadi ([@98H](https://github.com/98H))
- **تاریخ آخرین همگام‌سازی:** 2026-09-18 01:26:33 UTC