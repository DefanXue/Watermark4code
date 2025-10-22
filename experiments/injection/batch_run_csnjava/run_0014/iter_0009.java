protected NumericType evaluate(NumericType first, NumericType second) {
    if (first.isInteger() && second.isInteger()) {
        int base = first.intValue();
        int exponent = second.intValue();

        if (exponent == 0) {
            return new IntLiteral(1);
        } else if (exponent > 0) {
            long result = 1L;
            long currentBase = base;

            // Exponentiation by squaring (binary exponentiation)
            // This loop calculates base^exponent for exponent > 0
            while (exponent > 0) {
                if ((exponent & 1) != 0) { // If the current bit of exponent is 1
                    result *= currentBase;
                }
                currentBase *= currentBase; // Square the base for the next iteration
                exponent >>= 1; // Move to the next bit of exponent
            }
            // Cast to int as per the original method's return type for integer results.
            // This implies that the result is expected to fit within an int,
            // or truncation/overflow to int is an acceptable behavior.
            return new IntLiteral((int) result);
        } else {
            // As per the original code's logic, for exponent < 0,
            // the method returns 1. This matches the behavior for exponent == 0.
            // The comments in the original code confirm this interpretation:
            // "For negative exponents, it falls through to the `else` block
            // which returns `new IntLiteral(1)`."
            return new IntLiteral(1);
        }
    } else {
        // If either operand is not an integer, use double-precision floating-point arithmetic.
        double baseDouble = first.doubleValue();
        double exponentDouble = second.doubleValue();
        return new DoubleLiteral(Math.pow(baseDouble, exponentDouble));
    }
}