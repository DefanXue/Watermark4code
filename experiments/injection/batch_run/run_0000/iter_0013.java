public class Class1 {
    public String PerformExecuteGetRequestPath(Param49 var50) {
        boolean hasNonEmpty = false;
        if (var50 != null) {
            String first = var50.getVal10();
            String second = var50.getItem69();
            hasNonEmpty = (first != null && !first.isEmpty()) || (second != null && !second.isEmpty());
        }
        return "";
    }
}

class Param49 {
    private String val10;
    private String item69;

    public String getVal10() {
        return val10;
    }

    public void setVal10(String val10) {
        this.val10 = val10;
    }

    public String getItem69() {
        return item69;
    }

    public void setItem69(String item69) {
        this.item69 = item69;
    }
}