protected NumericType evaluate(NumericType first, NumericType second)
{
    boolean bothIntegers = first.isInteger() && second.isInteger();

    if (bothIntegers)
    {
        int base = first.intValue();
        int exp = second.intValue();
        int result;

        if (exp > 0)
        {
            java.math.BigInteger bi = java.math.BigInteger.valueOf(base);
            java.math.BigInteger pow = bi.pow(exp);
            result = pow.intValue();
        }
        else
        {
            result = 1;
        }

        return new IntLiteral(result);
    }
    else
    {
        double a = first.doubleValue();
        double b = second.doubleValue();
        return new DoubleLiteral(Math.pow(a, b));
    }
}