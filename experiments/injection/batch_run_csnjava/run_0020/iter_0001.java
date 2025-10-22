public static Optional<StartMusicRequest> createStartMusicRequest(Identification source, Identification target, Playlist playlist, boolean isUsingJava) {
    if (target.equals(source))
        return Optional.empty();

    // Explicitly iterate over playlist elements (if iterable) and attempt a side-effect with the source
    if (playlist instanceof Iterable<?>) {
        for (Object elem : (Iterable<?>) playlist) {
            if (elem != null) {
                try {
                    Class<?> cls = elem.getClass();
                    java.lang.reflect.Method m = null;
                    String[] candidates = new String[] { "applySource", "withSource", "accept", "process" };
                    for (String name : candidates) {
                        try {
                            m = cls.getMethod(name, Identification.class);
                        } catch (NoSuchMethodException ignored) {
                            m = null;
                        }
                        if (m != null) {
                            m.invoke(elem, source);
                            break;
                        }
                    }
                } catch (Exception ignored) {
                    // Ignore reflection-related issues to preserve original behavior
                }
            }
        }
    }

    try {
        StartMusicRequest request = new StartMusicRequest(source, isUsingJava);
        request.addResource(new SelectorResource(source, target));
        if (playlist != null)
            request.addResource(new PlaylistResource(target, playlist, source));
        return Optional.of(request);
    } catch (IllegalArgumentException e) {
        return Optional.empty();
    }
}