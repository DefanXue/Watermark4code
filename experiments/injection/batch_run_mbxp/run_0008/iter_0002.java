import java.util.HashMap;
import java.util.List;
import java.util.Map;

class SecondFrequent {
    public static String secondFrequent(List<String> inputValues) {
        Map<String, Integer> stringOccurrences = new HashMap<>();
        int currentIndex = 0;
        int listSize = inputValues.size();

        while (currentIndex < listSize) {
            String currentElement = inputValues.get(currentIndex);
            if (stringOccurrences.containsKey(currentElement)) {
                return currentElement;
            }
            stringOccurrences.put(currentElement, 1);
            currentIndex++;
        }
        return "";
    }
}