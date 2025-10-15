import tkinter as tk
from tkinter import ttk, messagebox
import json
from datetime import datetime
import os


class Vegetables:
    total_weight = 0
    total_price = 0

    def __init__(self, name, weight, price_per_kg):
        self.name = name
        self.weight = weight  # в граммах
        self.price_per_kg = price_per_kg  # цена за 1 кг
        self.price = (weight * price_per_kg) / 1000  # цена за указанный вес

        Vegetables.total_weight += weight
        Vegetables.total_price += self.price

    def __str__(self):
        return f"{self.__class__.__name__}: {self.name}, {self.weight}g, {self.price:.2f}rub (за кг: {self.price_per_kg}rub)"

    @classmethod
    def get_total_stats(cls):
        return cls.total_weight, cls.total_price

    def update(self, name, weight, price_per_kg):
        # Вычитаем старые значения
        Vegetables.total_weight -= self.weight
        Vegetables.total_price -= self.price

        # Обновляем значения
        self.name = name
        self.weight = weight
        self.price_per_kg = price_per_kg
        self.price = (weight * price_per_kg) / 1000

        # Добавляем новые значения
        Vegetables.total_weight += weight
        Vegetables.total_price += self.price


class Root(Vegetables):
    pass


class Leaf(Vegetables):
    pass


class Legumes(Vegetables):
    pass


class Carrot(Root):
    pass


class Beet(Root):
    pass


class Lettuce(Leaf):
    pass


class Spinach(Leaf):
    pass


class Chickpea(Legumes):
    pass


class Bean(Legumes):
    pass


class VegetableManager:
    def __init__(self):
        self.vegetables = []
        self.filename = r"C:\Users\kek13\Downloads\vegetables_data.json"
        self.window = tk.Tk()
        self.window.title("Управление овощами")
        self.window.geometry("700x450")

        # Загружаем данные при запуске
        self.load_data()
        self.setup_ui()

    def setup_ui(self):
        # Основная рамка
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Кнопки управления
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        ttk.Button(btn_frame, text="Добавить", command=self.add_dialog).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Редактировать", command=self.edit_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Удалить", command=self.delete_vegetable).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Общая статистика", command=self.show_stats).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Сохранить в файл", command=self.save_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Загрузить из файла", command=self.load_data).pack(side=tk.LEFT, padx=5)

        # Список овощей
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # Заголовки столбцов
        headers = ["Тип", "Название", "Вес (г)", "Цена за кг", "Стоимость"]
        for i, header in enumerate(headers):
            ttk.Label(list_frame, text=header, font=('Arial', 10, 'bold')).grid(row=0, column=i, padx=2, pady=2)

        # Treeview для отображения данных в таблице
        self.tree = ttk.Treeview(list_frame, columns=headers, show="headings", height=12)
        for i, header in enumerate(headers):
            self.tree.heading(header, text=header)
            self.tree.column(header, width=120)

        self.tree.grid(row=1, column=0, columnspan=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Полоса прокрутки
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.grid(row=1, column=5, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Обновление списка
        self.update_treeview()

        # Настройка расширения
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(1, weight=1)

    def add_dialog(self):
        self._open_editor()

    def edit_dialog(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите овощ для редактирования")
            return

        item = selection[0]
        index = self.tree.index(item)
        vegetable = self.vegetables[index]
        self._open_editor(vegetable, index)

    def _open_editor(self, vegetable=None, index=None):
        # Создание диалогового окна
        dialog = tk.Toplevel(self.window)
        dialog.title("Редактор овощей" if vegetable else "Добавление овоща")
        dialog.geometry("350x200")
        dialog.transient(self.window)
        dialog.grab_set()

        # Переменные формы
        name_var = tk.StringVar(value=vegetable.name if vegetable else "")
        weight_var = tk.StringVar(value=str(vegetable.weight) if vegetable else "")
        price_per_kg_var = tk.StringVar(value=str(vegetable.price_per_kg) if vegetable else "")
        type_var = tk.StringVar(value=vegetable.__class__.__name__ if vegetable else "Carrot")

        # Элементы формы
        ttk.Label(dialog, text="Название:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(dialog, textvariable=name_var).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        ttk.Label(dialog, text="Вес (г):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(dialog, textvariable=weight_var).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        ttk.Label(dialog, text="Цена за кг (руб):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(dialog, textvariable=price_per_kg_var).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        ttk.Label(dialog, text="Тип:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        type_combo = ttk.Combobox(dialog, textvariable=type_var, state="readonly")
        type_combo['values'] = ['Carrot', 'Beet', 'Lettuce', 'Spinach', 'Chickpea', 'Bean']
        type_combo.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Метка для отображения рассчитанной стоимости
        calc_label = ttk.Label(dialog, text="Стоимость: -")
        calc_label.grid(row=4, column=0, columnspan=2, pady=5)

        # Функция автоматического расчета стоимости
        def calculate_price(*args):
            try:
                weight = int(weight_var.get()) if weight_var.get() else 0
                price_per_kg = float(price_per_kg_var.get()) if price_per_kg_var.get() else 0
                calculated_price = (weight * price_per_kg) / 1000
                calc_label.config(text=f"Стоимость: {calculated_price:.2f} руб")
            except:
                calc_label.config(text="Стоимость: -")

        # Привязываем автоматический пересчет при изменении
        weight_var.trace('w', calculate_price)
        price_per_kg_var.trace('w', calculate_price)

        # Фрейм с кнопками
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=10)

        def save_changes():
            try:
                name = name_var.get()
                weight = int(weight_var.get())
                price_per_kg = float(price_per_kg_var.get())
                veg_type = type_var.get()

                if not name or weight <= 0 or price_per_kg <= 0:
                    raise ValueError("Некорректные данные")

                class_map = {
                    'Carrot': Carrot,
                    'Beet': Beet,
                    'Lettuce': Lettuce,
                    'Spinach': Spinach,
                    'Chickpea': Chickpea,
                    'Bean': Bean
                }

                if vegetable:
                    vegetable.update(name, weight, price_per_kg)
                    # Обновляем класс если нужно
                    if vegetable.__class__.__name__ != veg_type:
                        self.vegetables[index] = class_map[veg_type](name, weight, price_per_kg)
                else:
                    new_veg = class_map[veg_type](name, weight, price_per_kg)
                    self.vegetables.append(new_veg)

                self.update_treeview()
                self.save_data()  # Автоматическое сохранение
                dialog.destroy()

            except ValueError as e:
                messagebox.showerror("Ошибка", f"Ошибка в данных: {e}")

        ttk.Button(btn_frame, text="Сохранить", command=save_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Отмена", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

        # Инициализируем расчет при открытии диалога
        calculate_price()

        dialog.columnconfigure(1, weight=1)
        dialog.rowconfigure(5, weight=1)

    def delete_vegetable(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Предупреждение", "Выберите овощ для удаления")
            return

        item = selection[0]
        index = self.tree.index(item)
        vegetable = self.vegetables[index]

        if messagebox.askyesno("Подтверждение", f"Удалить {vegetable.name}?"):
            # Вычитаем из общей статистики
            Vegetables.total_weight -= vegetable.weight
            Vegetables.total_price -= vegetable.price
            self.vegetables.pop(index)
            self.update_treeview()
            self.save_data()  # Автоматическое сохранение

    def show_stats(self):
        total_weight, total_price = Vegetables.get_total_stats()
        messagebox.showinfo("Общая статистика",
                            f"Общий вес: {total_weight}g\nОбщая стоимость: {total_price:.2f} руб")

    def update_treeview(self):
        # Очищаем treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Заполняем данными
        for veg in self.vegetables:
            self.tree.insert("", "end", values=(
                veg.__class__.__name__,
                veg.name,
                veg.weight,
                f"{veg.price_per_kg:.2f}",
                f"{veg.price:.2f}"
            ))

    def save_data(self):
        try:
            # Создаем папку если ее нет
            directory = os.path.dirname(self.filename)
            if not os.path.exists(directory):
                os.makedirs(directory)

            data = {
                "saved_at": datetime.now().isoformat(),
                "vegetables": []
            }

            for veg in self.vegetables:
                veg_data = {
                    "type": veg.__class__.__name__,
                    "name": veg.name,
                    "weight": veg.weight,
                    "price_per_kg": veg.price_per_kg
                }
                data["vegetables"].append(veg_data)

            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            messagebox.showinfo("Успех", f"Данные сохранены в файл: {self.filename}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить данные: {e}")

    def load_data(self):
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Сбрасываем статистику
            Vegetables.total_weight = 0
            Vegetables.total_price = 0
            self.vegetables.clear()

            class_map = {
                'Carrot': Carrot,
                'Beet': Beet,
                'Lettuce': Lettuce,
                'Spinach': Spinach,
                'Chickpea': Chickpea,
                'Bean': Bean
            }

            for veg_data in data["vegetables"]:
                veg_class = class_map.get(veg_data["type"])
                if veg_class:
                    vegetable = veg_class(
                        veg_data["name"],
                        veg_data["weight"],
                        veg_data["price_per_kg"]
                    )
                    self.vegetables.append(vegetable)

            self.update_treeview()
            messagebox.showinfo("Успех", f"Загружено {len(self.vegetables)} записей из файла: {self.filename}")

        except FileNotFoundError:
            messagebox.showinfo("Информация", "Файл с данными не найден. Будет создан новый при сохранении.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {e}")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = VegetableManager()
    app.run()