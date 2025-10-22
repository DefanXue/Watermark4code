import java.util.*;

class SecondFrequent {
    public static String secondFrequent(List<String> input) {
        Map<String, Integer> seen = new HashMap<>();
        for (String s : input) {
            if (seen.containsKey(s)) {
                return s;
            } else {
                seen.put(s, 1);
            }
        }
        return "";
    }
}