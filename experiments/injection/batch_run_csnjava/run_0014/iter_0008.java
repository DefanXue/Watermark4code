protected NumericType evaluate(NumericType first, NumericType second) {
    if (first.isInteger() && second.isInteger()) {
        int base = first.intValue();
        int exponent = second.intValue();

        if (exponent == 0) {
            return new IntLiteral(1);
        } else if (exponent > 0) {
            long result = 1L;
            long currentBase = base; // Use long for intermediate calculations to prevent overflow for currentBase * currentBase

            // Exponentiation by squaring (binary exponentiation)
            while (exponent > 0) {
                if ((exponent & 1) != 0) { // If the current bit is 1
                    result *= currentBase;
                }
                currentBase *= currentBase; // Square the base
                exponent >>= 1; // Move to the next bit
            }
            // The problem statement implies the result should fit in an int for integer base/exponent.
            // If the original code's `(int) result` cast is acceptable for potential overflow
            // then this is fine. If `result` could exceed Integer.MAX_VALUE and still be
            // considered an integer in the domain, then `NumericType` might need to handle
            // larger integers, but based on `IntLiteral` return, it's expected to fit.
            return new IntLiteral((int) result);
        } else {
            // For negative integer exponents, the result is 1/base^|exp|.
            // Since the return type for integer path is IntLiteral,
            // and 1/X for X > 1 is not an integer, the original code
            // only handles exp > 0 and exp == 0 returning 1.
            // For negative exponents, it falls through to the `else` block
            // which returns `new IntLiteral(1)`.
            // The problem states `if (exp > 0)` and `else { return new IntLiteral(1); }`
            // This means for exp <= 0, it returns 1.
            // My refactoring maintains this: `exponent == 0` returns 1,
            // and `exponent < 0` falls into this `else` block returning 1.
            return new IntLiteral(1);
        }
    } else {
        // If either is not an integer, use double precision Math.pow
        double baseDouble = first.doubleValue();
        double exponentDouble = second.doubleValue();
        return new DoubleLiteral(Math.pow(baseDouble, exponentDouble));
    }
}