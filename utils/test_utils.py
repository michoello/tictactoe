def roughlyEqual(m1, m2, digs=2):
    if all(
        [
            round(a, digs) == round(b, digs)
            for row_a, row_b in zip(m1, m2)
            for a, b in zip(row_a, row_b)
        ]
    ):
        return True
    print(f"Arrays not equal:\n{m1}\n{m2}")
    row = 0
    for row_a, row_b in zip(m1, m2):
        col = 0
        for a, b in zip(row_a, row_b):
            a, b = round(a, digs), round(b, digs)
            if a != b:
                print(f"{a} != {b} (row {row} col {col})")
            col += 1
        row += 1
    return False
