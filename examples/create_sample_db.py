import sqlite3
import os

# 确保脚本在examples目录中执行
db_path = os.path.join(os.path.dirname(__file__), 'sample_database.db')

# 创建连接
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 创建销售数据表
cursor.execute('''
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    product TEXT,
    region TEXT,
    amount REAL,
    profit REAL,
    customer_age INTEGER,
    customer_gender TEXT,
    rating REAL
)
''')

# 删除旧数据
cursor.execute('DELETE FROM sales')

# 插入样本数据
sales_data = [
    ('2023-01-01', '电子产品', '华东', 10200, 1800, 25, '男', 4.5),
    ('2023-01-05', '家居用品', '华南', 5500, 1100, 32, '女', 4.2),
    ('2023-01-10', '服装', '华北', 6800, 1500, 28, '女', 4.7),
    ('2023-01-15', '电子产品', '西南', 12500, 2200, 41, '男', 3.9),
    ('2023-01-20', '食品', '华中', 3200, 650, 35, '女', 4.3),
    ('2023-02-01', '电子产品', '华南', 11800, 2050, 33, '男', 4.6),
    ('2023-02-05', '家居用品', '华东', 6200, 1250, 27, '女', 4.4),
    ('2023-02-10', '食品', '华北', 2800, 580, 45, '男', 3.8),
    ('2023-02-15', '服装', '西南', 5300, 1150, 31, '女', 4.2),
    ('2023-02-20', '电子产品', '华中', 9800, 1750, 38, '男', 4.5),
    ('2023-03-01', '服装', '华东', 7500, 1650, 30, '女', 4.3),
    ('2023-03-05', '电子产品', '华南', 13200, 2450, 42, '男', 4.7),
    ('2023-03-10', '食品', '华北', 3500, 720, 36, '女', 4.1),
    ('2023-03-15', '家居用品', '西南', 5800, 1200, 29, '男', 4.2),
    ('2023-03-20', '服装', '华中', 6900, 1480, 34, '女', 4.5)
]

cursor.executemany('''
INSERT INTO sales (date, product, region, amount, profit, customer_age, customer_gender, rating)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
''', sales_data)

# 创建产品信息表
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    category TEXT,
    price REAL,
    cost REAL,
    inventory INTEGER
)
''')

# 删除旧数据
cursor.execute('DELETE FROM products')

# 插入产品数据
product_data = [
    ('智能手机', '电子产品', 4999, 3000, 120),
    ('笔记本电脑', '电子产品', 8999, 5500, 85),
    ('耳机', '电子产品', 999, 400, 200),
    ('沙发', '家居用品', 3999, 2000, 30),
    ('餐桌', '家居用品', 2499, 1200, 45),
    ('床垫', '家居用品', 1999, 800, 60),
    ('T恤', '服装', 199, 50, 500),
    ('牛仔裤', '服装', 299, 100, 400),
    ('夹克', '服装', 699, 250, 300),
    ('面包', '食品', 15, 5, 1000),
    ('牛奶', '食品', 12, 6, 800),
    ('水果', '食品', 20, 8, 500)
]

cursor.executemany('''
INSERT INTO products (name, category, price, cost, inventory)
VALUES (?, ?, ?, ?, ?)
''', product_data)

# 创建地区表
cursor.execute('''
CREATE TABLE IF NOT EXISTS regions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    population INTEGER,
    gdp_per_capita REAL,
    store_count INTEGER
)
''')

# 删除旧数据
cursor.execute('DELETE FROM regions')

# 插入地区数据
region_data = [
    ('华东', 35000000, 85000, 120),
    ('华南', 28000000, 92000, 95),
    ('华北', 32000000, 78000, 110),
    ('西南', 18000000, 65000, 75),
    ('华中', 25000000, 70000, 90),
    ('西北', 12000000, 60000, 50)
]

cursor.executemany('''
INSERT INTO regions (name, population, gdp_per_capita, store_count)
VALUES (?, ?, ?, ?)
''', region_data)

# 提交更改并关闭连接
conn.commit()
conn.close()

print(f"示例数据库创建成功：{db_path}")
print("创建了3个表:")
print("- sales: 销售数据表 (15条记录)")
print("- products: 产品信息表 (12条记录)")
print("- regions: 地区信息表 (6条记录)") 