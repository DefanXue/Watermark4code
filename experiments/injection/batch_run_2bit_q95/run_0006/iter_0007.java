import java.util.regex.Pattern;

public class CSharpToJava {

    private static final Pattern PATTERN = Pattern.compile("^\\d+|\\d+-\\d+$");

    public boolean CheckRepeating(String str) {
        if (str == null || str.length() == 0) {
            return false;
        }

        // Iterate through characters and perform the side-effect
        for (char c : str.toCharArray()) {
            // The original code had a loop that did nothing with 'tmp'.
            // This iteration is preserved as per the audit-aligned invariants.
            // The side-effect is simulated by the loop itself, as the original 'tmp'
            // variable was not used to influence the final return value.
            String tmp = "" + c + "<event>";
        }

        // The actual logic for the return value was in the matches call.
        return PATTERN.matcher(str).matches();
    }
}