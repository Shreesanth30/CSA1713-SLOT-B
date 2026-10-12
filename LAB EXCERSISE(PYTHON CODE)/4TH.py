# Crypt-arithmetic solver for addition puzzles, such as:
# SEND + MORE = MONEY

def solve_cryptarithm(addends, result):
    words = addends + [result]
    letters = set("".join(words))

    if len(letters) > 10:
        print("No solution: more than 10 different letters.")
        return

    leading_letters = {word[0] for word in words if len(word) > 1}
    assignment = {}
    used_digits = set()
    solution_count = 0

    def check():
        values = {
            word: int("".join(str(assignment[letter]) for letter in word))
            for word in words
        }

        if sum(values[word] for word in addends) == values[result]:
            return values
        return None

    def search(remaining_letters):
        nonlocal solution_count

        if not remaining_letters:
            values = check()
            if values is not None:
                solution_count += 1
                print(f"\nSolution {solution_count}:")
                print("Letter assignments:", assignment)
                print(" + ".join(str(values[word]) for word in addends),
                      "=", values[result])
            return

        letter = remaining_letters[0]

        for digit in range(10):
            if digit in used_digits:
                continue
            if digit == 0 and letter in leading_letters:
                continue

            assignment[letter] = digit
            used_digits.add(digit)

            search(remaining_letters[1:])

            used_digits.remove(digit)
            del assignment[letter]

    search(sorted(letters))

    if solution_count == 0:
        print("No solution found.")


if __name__ == "__main__":
    # Edit these words to solve another addition puzzle.
    solve_cryptarithm(["SEND", "MORE"], "MONEY")
