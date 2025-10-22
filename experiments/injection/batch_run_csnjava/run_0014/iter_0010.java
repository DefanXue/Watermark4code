protected NumericType evaluate(NumericType firstOperand, NumericType secondOperand) {
    // If both operands are integers, perform integer exponentiation.
    if (firstOperand.isInteger() && secondOperand.isInteger()) {
        int base = firstOperand.intValue();
        int exponent = secondOperand.intValue();

        if (exponent == 0) {
            // base^0 is 1.
            return new IntLiteral(1);
        } else if (exponent > 0) {
            // Calculate base^exponent using exponentiation by squaring.
            long result = 1L;
            long currentBase = base; // Use long to prevent overflow during intermediate squaring

            while (exponent > 0) {
                if ((exponent & 1) != 0) { // If the current bit of exponent is 1
                    result *= currentBase;
                }
                currentBase *= currentBase; // Square the base for the next iteration
                exponent >>= 1;             // Move to the next bit of exponent
            }
            // Cast to int as per the original method's return type for integer results.
            // This implies that the result is expected to fit within an int,
            // or truncation/overflow to int is an acceptable behavior.
            return new IntLiteral((int) result);
        } else {
            // For negative exponents, the original logic returns 1.
            // This matches the behavior for exponent == 0.
            return new IntLiteral(1);
        }
    } else {
        // If either operand is not an integer, use double-precision floating-point arithmetic.
        double baseDouble = firstOperand.doubleValue();
        double exponentDouble = secondOperand.doubleValue();
        return new DoubleLiteral(Math.pow(baseDouble, exponentDouble));
    }
}