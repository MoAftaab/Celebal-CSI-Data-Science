# Triangle Pattern Programs

This folder contains Python programs that demonstrate different triangle patterns using asterisk (*) characters.

## Files Included

1. `lower_triangular.py`: Generates a lower triangular pattern
2. `upper_triangular.py`: Generates an upper triangular pattern
3. `pyramid.py`: Generates a pyramid pattern

## Pattern Examples

### Lower Triangular Pattern
```
*
* *
* * *
* * * *
* * * * *
```

### Upper Triangular Pattern
```
* * * * *
* * * *
* * *
* *
*
```

### Pyramid Pattern
```
        *
      * * *
    * * * * *
  * * * * * * *
* * * * * * * * *
```

## Usage

Each Python file can be executed independently:

```bash
python lower_triangular.py
python upper_triangular.py
python pyramid.py
```

When you run any of the programs, you will be prompted to enter the number of rows for the pattern. Input must be a positive integer.

## Functions

Each file contains a main function to print the specific pattern:

- `print_lower_triangular(rows)`: Prints a lower triangular pattern with the specified number of rows
- `print_upper_triangular(rows)`: Prints an upper triangular pattern with the specified number of rows
- `print_pyramid(rows)`: Prints a pyramid pattern with the specified number of rows

## Input Validation

All programs include input validation to ensure:
- The input is a valid integer
- The input is a positive number

You can also modify the number of rows in each pattern by changing the `rows` variable in the respective files. 