import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

#  Ввод размерности
while True:
    try:
        n = int(input("Введите размерность матрицы - n(от 4 до 50):\n"))
        if 4 <= n <= 50:
            break
        else:
            print("Недопустимая размерность")
    except ValueError:
        print("Недопустимая размерность")
ndiv2 = n // 2

#  Ввод числа k
while True:
    try:
        k = int(input("Введите K (целое число):\n"))
        break
    except ValueError:
        print("Введенно некорректное значение")

start = time.time()

#  Создание матрицы А
A = np.random.randint(-10, 10, (n, n))
print(f"Сгенерированная матрица А: \n{A}")

#  Иницилизация матрицы F
F = np.copy(A)
print(f"\nМатрица F до преобразований: \n{F}")

#  Подсчет нулевых элементов в E
count1, count2 = 0, 0
for i in range(n):
    for j in range(n):
        if j > (ndiv2 - (n - 1) % 2) and i > ndiv2:
            if F[i][j] == 0 and j % 2 != 0:
                count1 += 1
            elif F[i][j] < 0 and i % 2 == 0:
                count2 += 1

#  Преобразование матрицы F согласно условию
if count1 > count2:  # Если верно, то меняем местами С и В симметрично
    for i in range(ndiv2):
        F[i] = F[i][::-1]
else:  # Иначе, меняем C и E несимметрично
    for i in range(n):
        for j in range(n):
            if j > (ndiv2 - (n - 1) % 2) and i < ndiv2:
                if n % 2 == 0:
                    F[i][j] = F[i + ndiv2][j]
                    F[i + ndiv2][j] = A[i][j]
                else:
                    F[i][j] = F[i + ndiv2 + 1][j]
                    F[i + ndiv2 + 1][j] = A[i][j]
print(f"\nМатрица F после преобразований: \n{F}")

#  Вычисление согласно условию
if np.linalg.det(A) > np.trace(F):  # Если верно, то вычисляем A*AT – K * FТ
    if np.linalg.det(A) == 0 or np.linalg.det(F) == 0:  # Проверка определителей матриц
        print("\nМатрица F и(или) матрица A вырождена")
    else:
        AT = A.T
        print(f"\nТранспонированная матрица А: \n{AT}")
        AAT = A * AT
        print(f"\nМатрица А умноженная на транспонированную матрицу А: \n{AAT}")
        FT = F.T
        print(f"\nТранспонированная матрица F: \n{FT}")
        KFT = k * FT
        print(f"\nЧисло К умноженное на транспонированную матрицу F: \n{KFT}")
        print(f"\nФинальный результат: \n{AAT - KFT}")
else:  # Иначе, вычисляем (AТ +G-F^-1)*K
    if np.linalg.det(A) == 0 or np.linalg.det(F) == 0:  # Проверка определителей матриц
        print("Матрица F и(или) матрица A вырождена")
    else:
        AT = A.T
        print(f"\nТранспонированная матрица А: \n{AT}")
        G = np.tril(A)
        print(f"\nНижняя треугольная матрица G: \n{G}")
        invF = np.linalg.inv(F)
        print(f"\nОбратная матрица F: \n{invF}")
        Fin = (AT + G - invF) * k
        print(f"\nФинальный результат: \n{(AT + G - invF) * k}")
finish = time.time()
print(f"Время работы программы: {finish - start} секунд")

fig, ax = plt.subplots()                            #matplotlib
ax.set(xlabel='column number', ylabel='value')
for i in range(n):
    for j in range(n):
        plt.bar(i, Fin[i][j])
plt.show()

fig, ax = plt.subplots()
ax.set(xlabel='column number', ylabel='value')
ax.grid()
for j in range(n):
    ax.plot([i for i in range(n)], Fin[j][::])
plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.set(xlabel='x', ylabel='y', zlabel='z')
for i in range(n):
    plt.plot([j for j in range(n)], Fin[i][::], i)
plt.show()



sns.heatmap(data = F, annot = True)                 #seaborn
plt.xlabel('column number')
plt.ylabel('row number')
plt.show()

sns.boxplot(data = F)
plt.xlabel('column number')
plt.ylabel('value')
plt.show()

sns.lineplot(data = F)
plt.xlabel('column number')
plt.ylabel('value')
plt.show()
