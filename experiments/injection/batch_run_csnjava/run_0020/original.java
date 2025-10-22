public static Optional<StartMusicRequest> createStartMusicRequest(Identification source, Identification target, Playlist playlist, boolean isUsingJava) {
        if (target.equals(source))
            return Optional.empty();
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