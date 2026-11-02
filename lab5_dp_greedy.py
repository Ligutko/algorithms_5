"""Лабораторна робота № 5. Реалізація алгоритмів дискретної оптимізації.

Варіант 2.
Завдання 1 (динамічне програмування): оптимальне розрізання стрижня,
мінімальна довжина відрізка l_min = 2.
Завдання 2 (жадібний алгоритм): черга 3D-принтера, час друку групи моделей
дорівнює мінімальному часу друку серед моделей групи.

Запуск: python lab5_dp_greedy.py [rod | printer | check]
Без аргументу виконуються всі три розділи.
"""
import itertools
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MIN_PIECE = 2


# ───────────────────── Завдання 1. Розрізання стрижня ─────────────────────
#
# best(n) - найбільший прибуток від стрижня довжини n. Перша частина має
# довжину i (i >= MIN_PIECE), за неї отримують prices[i - 1], решта довжини
# n - i розрізається оптимально:
#     best(0) = 0,   best(n) = max по i від MIN_PIECE до n з
#                              prices[i - 1] + best(n - i).
# Якщо для n - i розв'язку немає (n - i = 1), значення i пропускається.

def validate_rod(length: int, prices: List[int], min_piece: int = MIN_PIECE) -> None:
    if length <= 0:
        raise ValueError("Довжина стрижня повинна бути > 0")
    if not prices:
        raise ValueError("Масив цін не може бути порожнім")
    if len(prices) != length:
        raise ValueError("Довжина масиву цін повинна відповідати довжині стрижня")
    if any(p <= 0 for p in prices):
        raise ValueError("Усі ціни повинні бути > 0")
    if length < min_piece:
        raise ValueError(f"Стрижень довжини {length} продати неможливо: "
                         f"мінімальна довжина відрізка {min_piece}")


def build_result(length: int, max_profit: int, first) -> Dict:
    cuts, n = [], length
    while n > 0:
        cuts.append(first[n])
        n -= first[n]
    cuts.sort(reverse=True)
    return {"max_profit": max_profit, "cuts": cuts,
            "number_of_cuts": len(cuts) - 1}


def rod_cutting_memo(length: int, prices: List[int], min_piece: int = MIN_PIECE,
                     stats: Optional[Dict] = None) -> Dict:
    validate_rod(length, prices, min_piece)
    memo = {0: 0}
    first = {}

    def solve(n: int) -> Optional[int]:
        if stats is not None:
            stats["calls"] += 1
        if n in memo:
            return memo[n]
        best = None
        for i in range(min_piece, n + 1):
            rest = solve(n - i)
            if rest is None:
                continue
            profit = prices[i - 1] + rest
            if best is None or profit > best:
                best, first[n] = profit, i
        memo[n] = best
        return best

    max_profit = solve(length)
    return build_result(length, max_profit, first)


def rod_cutting_table(length: int, prices: List[int], min_piece: int = MIN_PIECE,
                      stats: Optional[Dict] = None) -> Dict:
    validate_rod(length, prices, min_piece)
    best = [None] * (length + 1)
    best[0] = 0
    first = [0] * (length + 1)
    for n in range(min_piece, length + 1):
        for i in range(min_piece, n + 1):
            rest = best[n - i]
            if rest is None:
                continue
            profit = prices[i - 1] + rest
            if best[n] is None or profit > best[n]:
                best[n], first[n] = profit, i
    if stats is not None:
        stats["best"], stats["first"] = best, first
    return build_result(length, best[length], first)


def rod_cutting_naive(n: int, prices: List[int], stats: Dict) -> Optional[int]:
    """Та сама рекурсія без запам'ятовування; лічильник викликів у stats."""
    stats["calls"] += 1
    if n == 0:
        return 0
    best = None
    for i in range(MIN_PIECE, n + 1):
        rest = rod_cutting_naive(n - i, prices, stats)
        if rest is not None and (best is None or prices[i - 1] + rest > best):
            best = prices[i - 1] + rest
    return best


def all_cuttings(n: int, min_piece: int = MIN_PIECE):
    """Усі способи розрізання довжини n на частини не коротші за min_piece."""
    if n == 0:
        yield []
        return
    for i in range(min_piece, n + 1):
        for rest in all_cuttings(n - i, min_piece):
            yield [i] + rest


def brute_force_best(length: int, prices: List[int], min_piece: int = MIN_PIECE) -> int:
    return max(sum(prices[i - 1] for i in cut)
               for cut in all_cuttings(length, min_piece))


def print_result(title: str, result: Dict) -> None:
    print(f"\n{title}")
    print(f"Максимальний прибуток: {result['max_profit']}")
    print(f"Розрізи: {result['cuts']}")
    print(f"Кількість розрізів: {result['number_of_cuts']}")


def run_tests() -> None:
    test_cases = [
        {"length": 5, "prices": [2, 5, 7, 8, 10], "name": "Базовий випадок"},
        {"length": 3, "prices": [1, 3, 8], "name": "Оптимально не різати"},
        {"length": 4, "prices": [3, 5, 6, 7], "name": "Рівномірні розрізи"},
        {"length": 2, "prices": [1, 3], "name": "Мінімальний стрижень"},
        {"length": 10, "prices": [1, 5, 8, 9, 10, 17, 17, 20, 24, 30],
         "name": "Довгий стрижень"},
    ]
    print("=" * 60)
    print("ЗАВДАННЯ 1. Оптимальне розрізання стрижня (динамічне програмування)")
    print("=" * 60)
    for test in test_cases:
        print(f"\nТест: {test['name']}")
        print(f"Довжина стрижня: {test['length']}")
        print(f"Ціни: {test['prices']}")
        memo_result = rod_cutting_memo(test["length"], test["prices"])
        print_result("Результат мемоізації:", memo_result)
        table_result = rod_cutting_table(test["length"], test["prices"])
        print_result("Результат табуляції:", table_result)
        assert memo_result == table_result, "Результати двох методів не збігаються"
        assert 1 not in memo_result["cuts"], "У розрізах є відрізок довжини 1"
        print("\nПеревірка пройшла успішно!")


def print_dp_table(length: int, prices: List[int]) -> None:
    stats: Dict = {}
    rod_cutting_table(length, prices, stats=stats)
    best, first = stats["best"], stats["first"]
    print("\n1. Таблиця табуляції")
    print(f"Довжина стрижня: {length}, ціни: {prices}")
    print(f"{'n':>2}  {'кандидати (i: прибуток)':<34}{'best[n]':>8}{'first[n]':>9}")
    for n in range(length + 1):
        if n == 0:
            print(f"{0:>2}  {'база':<34}{0:>8}{'-':>9}")
            continue
        kandydaty = [f"{i}: {prices[i - 1] + best[n - i]}"
                     for i in range(MIN_PIECE, n + 1) if best[n - i] is not None]
        tekst = ", ".join(kandydaty) if kandydaty else "розв'язку немає"
        b = best[n] if best[n] is not None else "-"
        f = first[n] if best[n] is not None else "-"
        print(f"{n:>2}  {tekst:<34}{b:>8}{f:>9}")


def print_calls(length: int, prices: List[int]) -> None:
    naive, memo = {"calls": 0}, {"calls": 0}
    p_naive = rod_cutting_naive(length, prices, naive)
    p_memo = rod_cutting_memo(length, prices, stats=memo)["max_profit"]
    print(f"\n2. Кількість викликів рекурсії, довжина стрижня {length}")
    print(f"Без запам'ятовування: прибуток {p_naive}, викликів {naive['calls']}")
    print(f"З мемоізацією:        прибуток {p_memo}, викликів {memo['calls']}")
    print(f"Викликів менше у {naive['calls'] / memo['calls']:.0f} разів")


def print_errors() -> None:
    print("\n3. Некоректні вхідні дані")
    cases = [(0, [], "довжина 0"),
             (3, [], "порожній масив цін"),
             (3, [1, 2], "довжина масиву цін не збігається"),
             (3, [1, 0, 3], "ціна 0"),
             (1, [5], "довжина 1 при мінімумі 2")]
    for length, prices, opys in cases:
        try:
            rod_cutting_table(length, prices)
        except ValueError as e:
            print(f"{opys}: ValueError: {e}")


def print_rod_extra() -> None:
    print_dp_table(5, [2, 5, 7, 8, 10])
    print_calls(22, [p * 3 for p in range(1, 23)])
    print_errors()


# ───────────────────── Завдання 2. Черга 3D-принтера ─────────────────────
#
# Моделі впорядковуються за пріоритетом (1 - найвищий); при однаковому
# пріоритеті зберігається вхідний порядок. Партія - найдовший початок черги,
# що не перевищує max_volume за сумарним об'ємом і max_items за кількістю.
# Варіант 2: партія друкується стільки, скільки триває найшвидша модель
# партії. Незавершені моделі лишаються в черзі із залишком часу друку.

@dataclass
class PrintJob:
    id: str
    volume: float
    priority: int
    print_time: int


@dataclass
class PrinterConstraints:
    max_volume: float
    max_items: int


def validate_jobs(jobs: List[PrintJob], limits: PrinterConstraints) -> None:
    if limits.max_volume <= 0 or limits.max_items <= 0:
        raise ValueError("Обмеження принтера повинні бути > 0")
    if len({job.id for job in jobs}) != len(jobs):
        raise ValueError("Ідентифікатори завдань повинні бути унікальними")
    for job in jobs:
        if job.volume <= 0 or job.print_time <= 0:
            raise ValueError(f"Модель {job.id}: об'єм і час друку повинні бути > 0")
        if job.priority not in (1, 2, 3):
            raise ValueError(f"Модель {job.id}: пріоритет повинен бути 1, 2 або 3")
        if job.volume > limits.max_volume:
            raise ValueError(f"Модель {job.id} не вміщується в принтер навіть сама")


def take_batch(queue: List[PrintJob], limits: PrinterConstraints) -> List[PrintJob]:
    batch, volume = [], 0
    for job in queue:
        if batch and (volume + job.volume > limits.max_volume
                      or len(batch) == limits.max_items):
            break
        batch.append(job)
        volume += job.volume
    return batch


def optimize_printing(print_jobs: List[Dict], constraints: Dict,
                      time_rule=min) -> Dict:
    jobs = [PrintJob(**job) for job in print_jobs]
    limits = PrinterConstraints(**constraints)
    validate_jobs(jobs, limits)

    queue = sorted(jobs, key=lambda job: job.priority)
    left = {job.id: job.print_time for job in jobs}
    print_order, batches, total_time = [], [], 0
    while queue:
        batch = take_batch(queue, limits)
        step = time_rule(left[job.id] for job in batch)
        for job in batch:
            if job.id not in print_order:
                print_order.append(job.id)
            left[job.id] = max(0, left[job.id] - step)
        total_time += step
        batches.append({"jobs": [job.id for job in batch], "time": step})
        queue = [job for job in queue if left[job.id] > 0]
    return {"print_order": print_order, "total_time": total_time,
            "batches": batches}


CONSTRAINTS = {"max_volume": 300, "max_items": 2}

TEST_JOBS = [
    ("Тест 1 (однаковий пріоритет)", [
        {"id": "M1", "volume": 100, "priority": 1, "print_time": 120},
        {"id": "M2", "volume": 150, "priority": 1, "print_time": 90},
        {"id": "M3", "volume": 120, "priority": 1, "print_time": 150}]),
    ("Тест 2 (різні пріоритети)", [
        {"id": "M1", "volume": 100, "priority": 2, "print_time": 120},
        {"id": "M2", "volume": 150, "priority": 1, "print_time": 90},
        {"id": "M3", "volume": 120, "priority": 3, "print_time": 150}]),
    ("Тест 3 (перевищення обмежень)", [
        {"id": "M1", "volume": 250, "priority": 1, "print_time": 180},
        {"id": "M2", "volume": 200, "priority": 1, "print_time": 150},
        {"id": "M3", "volume": 180, "priority": 2, "print_time": 120}]),
    ("Тест 4 (п'ять моделей)", [
        {"id": "A1", "volume": 80, "priority": 1, "print_time": 100},
        {"id": "A2", "volume": 120, "priority": 1, "print_time": 60},
        {"id": "A3", "volume": 90, "priority": 2, "print_time": 150},
        {"id": "A4", "volume": 60, "priority": 3, "print_time": 40},
        {"id": "A5", "volume": 200, "priority": 2, "print_time": 80}]),
]


def print_printing(name: str, result: Dict) -> None:
    print(name + ":")
    print(f"Порядок друку: {result['print_order']}")
    for k, b in enumerate(result["batches"], start=1):
        print(f"Партія {k}: {b['jobs']}, {b['time']} хв")
    print(f"Загальний час: {result['total_time']} хвилин")


def test_printing_optimization() -> None:
    print("=" * 60)
    print("ЗАВДАННЯ 2. Оптимізація черги 3D-принтера (жадібний алгоритм)")
    print("=" * 60)
    print(f"Обмеження принтера: {CONSTRAINTS}\n")
    for i, (name, jobs) in enumerate(TEST_JOBS):
        if i:
            print()
        print_printing(name, optimize_printing(jobs, CONSTRAINTS))
    print("\nПорівняння з базовим правилом (час партії = максимум):")
    for name, jobs in TEST_JOBS:
        base = optimize_printing(jobs, CONSTRAINTS, time_rule=max)["total_time"]
        var = optimize_printing(jobs, CONSTRAINTS)["total_time"]
        print(f"{name.split(' (')[0]}: базове {base} хв, варіант 2 {var} хв")
    print("\nНекоректні вхідні дані:")
    bad = [("модель більша за принтер", [{"id": "X", "volume": 400,
                                          "priority": 1, "print_time": 10}]),
           ("пріоритет 4", [{"id": "X", "volume": 10, "priority": 4,
                             "print_time": 10}]),
           ("повторний id", [{"id": "X", "volume": 10, "priority": 1, "print_time": 10},
                             {"id": "X", "volume": 10, "priority": 1, "print_time": 20}])]
    for opys, jobs in bad:
        try:
            optimize_printing(jobs, CONSTRAINTS)
        except ValueError as e:
            print(f"{opys}: ValueError: {e}")


# ───────────────────────────── Перевірка ─────────────────────────────

# Результати методички для базового правила (мінімальна довжина відрізка 1,
# час партії - максимум): прибуток, розрізи, кількість розрізів.
ROD_BASE = [(5, [2, 5, 7, 8, 10], 12, [2, 2, 1], 2),
            (3, [1, 3, 8], 8, [3], 0),
            (4, [3, 5, 6, 7], 12, [1, 1, 1, 1], 3)]
PRINT_BASE = [(270, ["M1", "M2", "M3"]), (270, ["M2", "M1", "M3"]),
              (450, ["M1", "M2", "M3"])]


def simulate_minutes(print_jobs: List[Dict], constraints: Dict) -> int:
    """Покрокова симуляція: одна ітерація - одна хвилина друку."""
    jobs = sorted((PrintJob(**j) for j in print_jobs), key=lambda j: j.priority)
    limits = PrinterConstraints(**constraints)
    left = {j.id: j.print_time for j in jobs}
    minutes, active = 0, []
    while any(left.values()):
        if not active:
            volume = 0
            for j in (j for j in jobs if left[j.id] > 0):
                if active and (volume + j.volume > limits.max_volume
                               or len(active) == limits.max_items):
                    break
                active.append(j)
                volume += j.volume
        minutes += 1
        for j in active:
            left[j.id] -= 1
        if any(left[j.id] == 0 for j in active):
            active = []
    return minutes


def random_jobs(rng: random.Random):
    limits = {"max_volume": rng.randint(100, 250), "max_items": rng.randint(1, 4)}
    jobs = [{"id": f"J{k}", "volume": rng.randint(1, limits["max_volume"] // 2),
             "priority": rng.randint(1, 3), "print_time": rng.randint(1, 60)}
            for k in range(rng.randint(1, 8))]
    return jobs, limits


def print_check() -> None:
    print("=" * 60)
    print("ПЕРЕВІРКА")
    print("=" * 60)

    print("\n1. Базові результати методички відтворено")
    for length, prices, profit, cuts, n_cuts in ROD_BASE:
        for fn in (rod_cutting_memo, rod_cutting_table):
            r = fn(length, prices, min_piece=1)
            assert (r["max_profit"], r["cuts"], r["number_of_cuts"]) == \
                (profit, cuts, n_cuts), (fn.__name__, length)
        print(f"стрижень {length}: прибуток {profit}, розрізи {cuts}, "
              f"розрізів {n_cuts} (memo і table)")
    for (name, jobs), (total, order) in zip(TEST_JOBS, PRINT_BASE):
        r = optimize_printing(jobs, CONSTRAINTS, time_rule=max)
        assert (r["total_time"], r["print_order"]) == (total, order), name
        print(f"{name.split(' (')[0]}: {total} хв, порядок {order}")

    print("\n2. Стрижень: динамічне програмування проти повного перебору")
    rng = random.Random(5)
    sluchai = 1000
    for _ in range(sluchai):
        length = rng.randint(2, 14)
        prices = [rng.randint(1, 30) for _ in range(length)]
        memo = rod_cutting_memo(length, prices)
        table = rod_cutting_table(length, prices)
        assert memo == table
        assert memo["max_profit"] == brute_force_best(length, prices)
        assert sum(memo["cuts"]) == length and min(memo["cuts"]) >= MIN_PIECE
        assert sum(prices[c - 1] for c in memo["cuts"]) == memo["max_profit"]
        assert memo["number_of_cuts"] == len(memo["cuts"]) - 1
    print(f"випадкових наборів: {sluchai}, довжини 2-14, ціни 1-30")
    print("розбіжностей з повним перебором: 0")
    print("відрізків довжини 1 у відповідях: 0")

    print("\n3. Принтер: інваріанти і покрокова симуляція")
    sluchai = 2000
    for _ in range(sluchai):
        jobs, limits = random_jobs(rng)
        r = optimize_printing(jobs, limits)
        by_id = {j["id"]: j for j in jobs}
        used = {j["id"]: 0 for j in jobs}
        for b in r["batches"]:
            assert b["time"] > 0
            assert sum(by_id[i]["volume"] for i in b["jobs"]) <= limits["max_volume"]
            assert len(b["jobs"]) <= limits["max_items"]
            for i in b["jobs"]:
                used[i] += b["time"]
        assert all(used[i] == by_id[i]["print_time"] for i in used)
        assert r["total_time"] == sum(b["time"] for b in r["batches"])
        assert r["total_time"] >= max(j["print_time"] for j in jobs)
        prior = [by_id[i]["priority"] for i in r["print_order"]]
        assert prior == sorted(prior) and len(prior) == len(jobs)
        assert r["total_time"] == simulate_minutes(jobs, limits)
    print(f"випадкових наборів: {sluchai}, до 8 моделей")
    print("перевірено в кожному наборі:")
    print("  сумарний об'єм і кількість у кожній партії в межах обмежень")
    print("  час друку кожної моделі за всі партії дорівнює її print_time")
    print("  пріоритет у порядку друку не спадає")
    print("  загальний час збігається з покроковою симуляцією")
    print("розбіжностей: 0")


if __name__ == "__main__":
    rezhym = sys.argv[1] if len(sys.argv) > 1 else "all"
    if rezhym in ("rod", "all"):
        run_tests()
        print_rod_extra()
    if rezhym in ("printer", "all"):
        if rezhym == "all":
            print()
        test_printing_optimization()
    if rezhym in ("check", "all"):
        if rezhym == "all":
            print()
        print_check()
