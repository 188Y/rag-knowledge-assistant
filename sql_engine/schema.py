SCHEMA_DESCRIPTION = """
数据库包含以下表：

【表1：sales（销售记录）】
- id: INTEGER，主键
- region: TEXT，地区，取值只有：华东 / 华北 / 华南
- product: TEXT，产品名，取值示例：产品A / 产品B / 产品C
- amount: REAL，单笔销售金额（元）
- quantity: INTEGER，单笔销售数量
- sale_date: TEXT，销售日期，格式 YYYY-MM-DD（数据范围 2026-07 ~ 2026-09）

【表2：strategy（年度策略）】
- id: INTEGER，主键
- region: TEXT，地区（与 sales.region 对应）
- year: INTEGER，年份，取值示例：2026
- target_amount: REAL，该地区该年度的目标销售金额（元）
- focus_product: TEXT，重点产品

【表关联】
strategy.region = sales.region：需要对比"实际销售额 vs 目标"时，用 region 关联两张表。
"""