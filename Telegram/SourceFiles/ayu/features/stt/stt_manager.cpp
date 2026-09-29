// This is the source code of AyuGram for Desktop.
//
// We do not and cannot prevent the use of our code,
// but be respectful and credit the original author.
//
// Copyright @Radolyn, 2026

#include "ayu/features/stt/stt_manager.h"

#include "ayu/ayu_settings.h"
#include "base/debug_log.h"

#include <QtCore/QFile>
#include <QtCore/QStandardPaths>

#if defined(Q_OS_MAC)
#include "ayu/features/stt/platform/stt_mac.h"
#endif

#if defined(HAVE_WHISPER)
#include "ayu/features/stt/whisper_service.h"
#endif

#if defined(HAVE_WHISPER) && (defined(__x86_64__) || defined(_M_X64))
#define AYU_STT_CHECK_X86_CPU
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace Ayu::STT {

namespace {

constexpr std::array<const char*, 3> kModelFileNames = {
	"ggml-tiny.bin",
	"ggml-base.bin",
	"ggml-small.bin",
};

constexpr std::array<const char*, 3> kModelUrls = {
	"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
	"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
	"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
};

#if defined(AYU_STT_CHECK_X86_CPU)
void Cpuid(int leaf, std::array<uint32, 4> &regs) {
#if defined(_MSC_VER)
	int raw[4] = { 0 };
	__cpuidex(raw, leaf, 0);
	for (auto i = 0; i != 4; ++i) {
		regs[i] = static_cast<uint32>(raw[i]);
	}
#else
	__cpuid_count(leaf, 0, regs[0], regs[1], regs[2], regs[3]);
#endif
}

uint64 Xgetbv0() {
#if defined(_MSC_VER)
	return _xgetbv(0);
#else
	uint32 eax = 0, edx = 0;
	__asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
	return (uint64(edx) << 32) | eax;
#endif
}

// ggml is built with GGML_NATIVE=OFF, whose x86 baseline is
// AVX + AVX2 + FMA + F16C + BMI2. Running it on a CPU (or an emulator,
// e.g. Prism on Windows ARM before 24H2) without them is an illegal
// instruction crash, so refuse up front.
bool DetectCpuSupported() {
	auto regs = std::array<uint32, 4>();
	Cpuid(0, regs);
	if (regs[0] < 7) {
		return false;
	}
	Cpuid(1, regs);
	const auto ecx = regs[2];
	const auto fma = (ecx >> 12) & 1;
	const auto osxsave = (ecx >> 27) & 1;
	const auto avx = (ecx >> 28) & 1;
	const auto f16c = (ecx >> 29) & 1;
	if (!fma || !osxsave || !avx || !f16c) {
		return false;
	}
	// OS must save XMM and YMM state on context switch.
	if ((Xgetbv0() & 0x6) != 0x6) {
		return false;
	}
	Cpuid(7, regs);
	const auto ebx = regs[1];
	const auto avx2 = (ebx >> 5) & 1;
	const auto bmi2 = (ebx >> 8) & 1;
	return avx2 && bmi2;
}
#endif // AYU_STT_CHECK_X86_CPU

} // namespace

STTManager &STTManager::instance() {
	static STTManager self;
	return self;
}

QString STTManager::modelsDirectory() {
	const auto base = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
	return base + u"/whisper_models"_q;
}

QString STTManager::modelPath(const int modelType) {
	if (modelType < 0 || modelType >= static_cast<int>(kModelFileNames.size())) {
		return {};
	}
	return modelsDirectory() + u"/"_q + kModelFileNames[modelType];
}

QString STTManager::modelUrl(const int modelType) {
	if (modelType < 0 || modelType >= static_cast<int>(kModelUrls.size())) {
		return {};
	}
	return QString::fromUtf8(kModelUrls[modelType]);
}

bool STTManager::modelExists(const int modelType) {
	return QFile::exists(modelPath(modelType));
}

bool STTManager::cpuSupported() {
#if defined(AYU_STT_CHECK_X86_CPU)
	static const auto result = [] {
		const auto supported = DetectCpuSupported();
		if (!supported) {
			LOG(("STTManager: CPU lacks AVX2/FMA/F16C/BMI2, "
				"local Whisper disabled"));
		}
		return supported;
	}();
	return result;
#else
	return true;
#endif
}

void STTManager::requestPermission() {
#if defined(Q_OS_MAC)
	if (AyuSettings::getInstance().sttEngine() == STTEngine::AppleSpeech) {
		Mac::requestSpeechPermission();
	}
#endif
}

void STTManager::transcribe(const QString &filePath, std::function<void(QString)> callback) {
	const auto &settings = AyuSettings::getInstance();

#if defined(Q_OS_MAC)
	if (settings.sttEngine() == STTEngine::AppleSpeech) {
		Mac::transcribeFile(filePath, settings.sttLanguage(), std::move(callback));
		return;
	}
#endif

#if defined(HAVE_WHISPER)
	if (!cpuSupported()) {
		callback(QString());
		return;
	}
	const auto modelType = static_cast<int>(settings.whisperModelType());
	const auto path = modelPath(modelType);
	const auto language = settings.sttLanguage();

	WhisperService::instance().transcribeOnDemand(
		filePath,
		path,
		language,
		std::move(callback));
	return;
#endif

	// No engine available for the selected configuration: report failure
	// instead of silently dropping the callback (which would hang the spinner).
	callback(QString());
}

} // namespace Ayu::STT
