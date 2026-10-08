courses = ["A", "B", "C", "D", "E"]
slots = ["Slot1", "Slot2", "Slot3"]

course_names = {
    "A": "Artificial Intelligence",
    "B": "Database Systems",
    "C": "Computer Networks",
    "D": "Operating Systems",
    "E": "Software Engineering",
}

# Each pair in this list must be assigned different slots.
conflicts = [
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("B", "D"),
    ("C", "D"),
    ("C", "E"),
    ("D", "E"),
]

neighbors = {course: set() for course in courses}
for first, second in conflicts:
    neighbors[first].add(second)
    neighbors[second].add(first)


def is_consistent(course, slot, assignment):
    """Return True if this slot does not conflict with assigned neighbors."""
    for neighbor in neighbors[course]:
        if assignment.get(neighbor) == slot:
            return False
    return True


def plain_backtracking():
    assignment = {}
    attempts = 0
    backtracks = 0

    def search(index):
        nonlocal attempts, backtracks

        if index == len(courses):
            return True

        course = courses[index]

        for slot in slots:
            attempts += 1
            print(f"Try {course} = {slot}")

            if is_consistent(course, slot, assignment):
                assignment[course] = slot
                print(f"  Assign {course} = {slot}")

                if search(index + 1):
                    return True

                # This runs only if a later choice leads to a dead end.
                print(f"  Backtrack: undo {course} = {slot}")
                del assignment[course]
                backtracks += 1
            else:
                print(f"  Reject {course} = {slot} (conflict)")

        return False

    solved = search(0)
    return solved, assignment, attempts, backtracks


def forward_checking():
    assignment = {}
    domains = {course: slots.copy() for course in courses}
    attempts = 0
    backtracks = 0

    def search(index):
        nonlocal attempts, backtracks

        if index == len(courses):
            return True

        course = courses[index]

        # Use the original slot order, skipping values pruned from the domain.
        for slot in slots:
            if slot not in domains[course]:
                print(f"Skip {course} = {slot} (pruned from domain)")
                continue

            attempts += 1
            print(f"Try {course} = {slot}")

            if not is_consistent(course, slot, assignment):
                print(f"  Reject {course} = {slot} (conflict)")
                continue

            assignment[course] = slot
            print(f"  Assign {course} = {slot}")

            # Save domains so they can be restored if this choice fails.
            saved_domains = {
                variable: domain.copy()
                for variable, domain in domains.items()
            }

            domain_wipeout = False

            # Remove this slot from unassigned neighboring courses.
            for neighbor in neighbors[course]:
                if neighbor not in assignment and slot in domains[neighbor]:
                    domains[neighbor].remove(slot)
                    print(
                        f"  Forward check: remove {slot} from {neighbor}; "
                        f"{neighbor} domain = {domains[neighbor]}"
                    )

                    if not domains[neighbor]:
                        print(f"  Domain wipeout for {neighbor}")
                        domain_wipeout = True
                        break

            if not domain_wipeout and search(index + 1):
                return True

            print(f"  Backtrack: undo {course} = {slot}")
            del assignment[course]
            domains.clear()
            domains.update(saved_domains)
            backtracks += 1

        return False

    solved = search(0)
    return solved, assignment, attempts, backtracks


def print_result(title, result):
    solved, assignment, attempts, backtracks = result

    print(f"\n{title}")
    print("=" * len(title))

    if not solved:
        print("No valid timetable found.")
    else:
        for course in courses:
            print(
                f"{course} - {course_names[course]}: "
                f"{assignment[course]}"
            )

    print("Assignment attempts:", attempts)
    print("Backtracks:", backtracks)


print("PLAIN BACKTRACKING SEARCH")
plain_result = plain_backtracking()
print_result("Plain Backtracking Result", plain_result)

print("\n\nBACKTRACKING WITH FORWARD CHECKING")
fc_result = forward_checking()
print_result("Forward Checking Result", fc_result)
