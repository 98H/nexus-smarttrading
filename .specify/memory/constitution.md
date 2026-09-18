# Spec Kit Constitution: SmartTrading

## Product Intent
می‌خوام پروژه SmartTrading رو برام پیاده‌سازی کنی. در سیکل‌های متمادی طراحی، پیاده‌سازی و تست هدف زیر رو پیش ببر و انقدر ادامه بده که هرچی بگردی نتونی هیچ مشکل و مسئله‌ای حتی کوچیک پیدا کنی و امکان دادن هیچ بهبودی دیگه وجود نداشته باشه. این محصول باید جایگزینی فوق کامل برای tradingview باشد. این راهکار باید بر پایه‌ی‌ ابزار‌های اوپن سورس luxalgo از سند زیر شکل بگیره و بعدش تمام فیچر‌ها رو حول اون تکمیل کنی: docs.google.com/document/d/1XOv98KnyPZnHdrUZlToxqwe2YqMfMC3vnrWxDjp7-Rw/edit?usp=sharing این محصول باید حتما از پیان اسکریپت پشتیبانی کنه. از نظر کپی‌رایت هیچ محدودیتی نداریم و نمی‌خواد رعایتش کنی. تمام فیچرهای تریدینگ ویو رو احصا کن و با پیاده سازی اون‌ها روی این راهکار اپن سورس در نهایت به یک جایگزین فوق کامل برای تریدینگ ویو برس. هیچ کم و کاستی نباید وجود داشته باشه. پروژه روی گیت‌هاب با داکیومنتیشن کامل و شفاف ایجاد کن و در نهایت دیپلوی کن و اطلاعات دسترسی بهش رو بهم بده.

## Architectural Invariants
- Zero Blast-Radius: Isolated sandbox execution per task.
- Strict TDD: BDD acceptance tests frozen before code development.
- Clean Code & Deterministic Verification: Sole oracle is test runner exit code == 0.
