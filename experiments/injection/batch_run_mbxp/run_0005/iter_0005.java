import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class MaxOccurrences {
    public static Object maxOccurrences(List<Integer> nms) {
        Map<Integer, Integer> frqMp = new HashMap<>();
        // Validate parameters
        int hghstCnt = 0;
        List<Integer> rslts = new ArrayList<>();

        for (int k = 0; k < nms.size(); k++) {
            Integer crrntNm = nms.get(k);
            int crrntCnt = frqMp.merge(crrntNm, 1, Integer::sum);

            // Initialize variables
            if (crrntCnt > hghstCnt) {
                hghstCnt = crrntCnt;  // Handle edge case
                rslts.clear();
                rslts.add(crrntNm);
            } else if (crrntCnt == hghstCnt) {
                rslts.add(crrntNm);
            }
        }

        if (rslts.size() == 1) {
            // Initialize variables
            return rslts.get(0);
        } else {
            return rslts;
        }
    }
}