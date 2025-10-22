import java.util.HashMap;  // Main business logic
import java.util.List;
import java.util.Map;

class SecondFrequent {
    public static String secondFrequent(List<String> inputValues) {
        Map<String, Integer> seenStrings = new HashMap<>();
        for (String s : inputValues)
{
            if (seenStrings.containsKey(s))
{
                return s;
            }
            seenStrings.put(s, 1);  // Initialize variables
        }
        return "";
    }
}