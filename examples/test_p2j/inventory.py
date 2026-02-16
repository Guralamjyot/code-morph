"""
Inventory management system demonstrating classes, enums,
type hints, exception handling, sorting, and string formatting.
"""

from enum import Enum
from typing import Optional


class Category(Enum):
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FOOD = "food"
    BOOKS = "books"
    OTHER = "other"

    def is_perishable(self) -> bool:
        return self == Category.FOOD


class Product:
    def __init__(self, name: str, price: float, category: Category, sku: str, quantity: int = 0):
        if not name or name.strip() == "":
            raise ValueError("Name cannot be empty")
        if price < 0:
            raise ValueError("Price cannot be negative: " + str(price))
        if quantity < 0:
            raise ValueError("Quantity cannot be negative: " + str(quantity))
        self.name = name
        self.price = price
        self.category = category
        self.sku = sku
        self.quantity = quantity

    def get_total_value(self) -> float:
        return self.price * self.quantity

    def is_in_stock(self) -> bool:
        return self.quantity > 0

    def apply_discount(self, percent: float) -> float:
        if percent < 0 or percent > 100:
            raise ValueError("Discount must be 0-100, got " + str(percent))
        return self.price * (1 - percent / 100)

    def __str__(self) -> str:
        status = "In Stock" if self.is_in_stock() else "Out of Stock"
        return "[" + status + "] " + self.name + " ($" + str(round(self.price, 2)) + ") - " + self.category.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Product):
            return False
        return self.sku == other.sku

    def __hash__(self) -> int:
        return hash(self.sku)

    def __lt__(self, other: "Product") -> bool:
        return self.price < other.price


class Inventory:
    def __init__(self, name: str):
        self.name = name
        self.products: list[Product] = []

    def add_product(self, product: Product) -> None:
        for p in self.products:
            if p.sku == product.sku:
                raise ValueError("Product with SKU " + product.sku + " already exists")
        self.products.append(product)

    def remove_product(self, sku: str) -> Optional[Product]:
        for i in range(len(self.products)):
            if self.products[i].sku == sku:
                return self.products.pop(i)
        return None

    def find_by_name(self, query: str) -> list[Product]:
        results: list[Product] = []
        query_lower = query.lower()
        for p in self.products:
            if query_lower in p.name.lower():
                results.append(p)
        return results

    def find_by_category(self, category: Category) -> list[Product]:
        results: list[Product] = []
        for p in self.products:
            if p.category == category:
                results.append(p)
        return results

    def get_in_stock(self) -> list[Product]:
        results: list[Product] = []
        for p in self.products:
            if p.is_in_stock():
                results.append(p)
        return results

    def get_out_of_stock(self) -> list[Product]:
        results: list[Product] = []
        for p in self.products:
            if not p.is_in_stock():
                results.append(p)
        return results

    def restock(self, sku: str, quantity: int) -> bool:
        if quantity <= 0:
            raise ValueError("Restock quantity must be positive")
        for p in self.products:
            if p.sku == sku:
                p.quantity += quantity
                return True
        return False

    def sell(self, sku: str, quantity: int = 1) -> bool:
        if quantity <= 0:
            raise ValueError("Sell quantity must be positive")
        for p in self.products:
            if p.sku == sku:
                if p.quantity >= quantity:
                    p.quantity -= quantity
                    return True
                return False
        return False

    def get_total_value(self) -> float:
        total = 0.0
        for p in self.products:
            total += p.get_total_value()
        return total

    def get_total_products(self) -> int:
        return len(self.products)

    def get_total_items(self) -> int:
        total = 0
        for p in self.products:
            total += p.quantity
        return total

    def get_sorted_by_price(self, descending: bool = False) -> list[Product]:
        return sorted(self.products, reverse=descending)

    def get_sorted_by_name(self) -> list[Product]:
        return sorted(self.products, key=lambda p: p.name.lower())

    def get_min_price(self) -> float:
        if not self.products:
            return 0.0
        min_price = self.products[0].price
        for p in self.products:
            if p.price < min_price:
                min_price = p.price
        return min_price

    def get_max_price(self) -> float:
        if not self.products:
            return 0.0
        max_price = self.products[0].price
        for p in self.products:
            if p.price > max_price:
                max_price = p.price
        return max_price

    def get_summary(self) -> str:
        lines = []
        lines.append("Inventory: " + self.name)
        lines.append("  Products: " + str(self.get_total_products()))
        lines.append("  Items: " + str(self.get_total_items()))
        lines.append("  Total Value: $" + str(round(self.get_total_value(), 2)))
        lines.append("  Price Range: $" + str(round(self.get_min_price(), 2))
                      + " - $" + str(round(self.get_max_price(), 2)))

        # Count by category
        category_counts: dict[str, int] = {}
        for p in self.products:
            cat_name = p.category.value
            if cat_name in category_counts:
                category_counts[cat_name] += 1
            else:
                category_counts[cat_name] = 1

        if category_counts:
            lines.append("  Categories:")
            for cat_name, count in category_counts.items():
                lines.append("    " + cat_name + ": " + str(count))

        return "\n".join(lines)


def main():
    store = Inventory("TechStore")

    store.add_product(Product("Laptop", 999.99, Category.ELECTRONICS, "ELEC-001", 15))
    store.add_product(Product("USB Cable", 9.99, Category.ELECTRONICS, "ELEC-002", 200))
    store.add_product(Product("Python Cookbook", 45.50, Category.BOOKS, "BOOK-001", 30))
    store.add_product(Product("T-Shirt", 25.00, Category.CLOTHING, "CLTH-001", 50))
    store.add_product(Product("Energy Bar", 2.99, Category.FOOD, "FOOD-001", 100))
    store.add_product(Product("Headphones", 149.99, Category.ELECTRONICS, "ELEC-003", 0))

    print("=== Inventory Demo ===\n")
    print(store.get_summary())

    print("\nElectronics:")
    for p in store.find_by_category(Category.ELECTRONICS):
        print("  " + str(p))

    print("\nOut of stock:")
    for p in store.get_out_of_stock():
        print("  " + str(p))

    store.sell("ELEC-001", 3)
    store.restock("ELEC-003", 25)

    print("\nAfter sales/restock:")
    for p in store.products:
        if p.sku == "ELEC-001":
            print("  Laptop qty: " + str(p.quantity))
        if p.sku == "ELEC-003":
            print("  Headphones qty: " + str(p.quantity))

    print("\nBy price (cheapest first):")
    for p in store.get_sorted_by_price():
        print("  $" + str(round(p.price, 2)) + " - " + p.name)

    laptop = None
    for p in store.products:
        if p.sku == "ELEC-001":
            laptop = p
            break

    if laptop is not None:
        print("\nLaptop 20% off: $" + str(round(laptop.apply_discount(20), 2)))


if __name__ == "__main__":
    main()
