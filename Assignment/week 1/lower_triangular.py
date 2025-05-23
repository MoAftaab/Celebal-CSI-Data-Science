def print_lower_triangular(rows):
    for i in range(rows):

        for j in range(i + 1):
            print("*", end=" ")
        print()  

if __name__ == "__main__":
    try:
        rows = int(input("Enter number of rows for the lower triangular pattern: "))
        if rows <= 0:
            print("Please enter a positive integer")
        else:
            print("Lower Triangular Pattern:")
            print_lower_triangular(rows)
    except ValueError:
        print("Invalid input. Please enter a valid integer.") 