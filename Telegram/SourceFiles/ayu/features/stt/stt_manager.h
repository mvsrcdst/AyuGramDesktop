// This is the source code of AyuGram for Desktop.
//
// We do not and cannot prevent the use of our code,
// but be respectful and credit the original author.
//
// Copyright @Radolyn, 2026

#pragma once

#include <functional>
#include <QtCore/QString>

namespace Ayu::STT {

class STTManager {
public:
	static STTManager &instance();
	void transcribe(const QString &filePath, std::function<void(QString)> callback);
	static void requestPermission();
	[[nodiscard]] static QString modelPath(int modelType);
	[[nodiscard]] static QString modelUrl(int modelType);
	[[nodiscard]] static bool modelExists(int modelType);
	[[nodiscard]] static QString modelsDirectory();
	// False when the CPU can't run the bundled ggml build (Whisper only).
	[[nodiscard]] static bool cpuSupported();

private:
	STTManager() = default;
};

} // namespace Ayu::STT
