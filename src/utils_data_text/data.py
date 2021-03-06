"""Contains data generator for orgnaizing various audio data preprocessing
pipeline and offering data reader interface of PaddlePaddle requirements.
"""

import random
import numpy as np
import paddle
import paddle.fluid as fluid
from threading import local
from data_utils.utility import read_manifest
from data_utils.augmentor.augmentation import AugmentationPipeline
from data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from data_utils.speech import SpeechSegment
from data_utils.normalizer import FeatureNormalizer


class DataGenerator(object):
    """
    DataGenerator provides basic audio data preprocessing pipeline, and offers
    data reader interfaces of PaddlePaddle requirements.

    :param vocab_filepath: Vocabulary filepath for indexing tokenized
                           transcripts.
    :type vocab_filepath: str
    :param mean_std_filepath: File containing the pre-computed mean and stddev.
    :type mean_std_filepath: None|str
    :param augmentation_config: Augmentation configuration in json string.
                                Details see AugmentationPipeline.__doc__.
    :type augmentation_config: str
    :param max_duration: Audio with duration (in seconds) greater than
                         this will be discarded.
    :type max_duration: float
    :param min_duration: Audio with duration (in seconds) smaller than
                         this will be discarded.
    :type min_duration: float
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :type stride_ms: float
    :param window_ms: Window size (in milliseconds) for generating frames.
    :type window_ms: float
    :param use_dB_normalization: Whether to normalize the audio to -20 dB
                                before extracting the features.
    :type use_dB_normalization: bool
    :param random_seed: Random seed.
    :type random_seed: int
    :param keep_transcription_text: If set to True, transcription text will
                                    be passed forward directly without
                                    converting to index sequence.
    :type keep_transcription_text: bool
    :param place: The place to run the program.
    :type place: CPUPlace or CUDAPlace
    :param is_training: If set to True, generate text data for training,
                        otherwise,  generate text data for infer.
    :type is_training: bool
    """

    def __init__(self,
                 vocab_filepath,
                 mean_std_filepath,
                 augmentation_config='{}',
                 max_duration=float('inf'),
                 min_duration=0.0,
                 stride_ms=10.0,
                 window_ms=20.0,
                 use_dB_normalization=True,
                 random_seed=0,
                 keep_transcription_text=False,
                 place=paddle.CPUPlace(),
                 is_training=True):
        self._max_duration = max_duration
        self._min_duration = min_duration
        self._normalizer = FeatureNormalizer(mean_std_filepath)
        self._augmentation_pipeline = AugmentationPipeline(augmentation_config=augmentation_config,
                                                           random_seed=random_seed)
        self._speech_featurizer = SpeechFeaturizer(vocab_filepath=vocab_filepath,
                                                   stride_ms=stride_ms,
                                                   window_ms=window_ms,
                                                   use_dB_normalization=use_dB_normalization)
        self._rng = random.Random(random_seed)
        self._keep_transcription_text = keep_transcription_text
        self.epoch = 0
        self._is_training = is_training
        # for caching tar files info
        self._local_data = local()
        self._local_data.tar2info = {}
        self._local_data.tar2object = {}
        self._place = place

    def process_utterance(self, audio_file, transcript):
        """??????????????????????????????????????????????????????

        :param audio_file: ??????????????????????????????????????????
        :type audio_file: str | file
        :param transcript: ?????????????????????
        :type transcript: str
        :return: ????????????????????????????????????????????????????????????????????????ID
        :rtype: tuple of (2darray, list)
        """
        speech_segment = SpeechSegment.from_file(audio_file, transcript)
        self._augmentation_pipeline.transform_audio(speech_segment)
        specgram, transcript_part = self._speech_featurizer.featurize(speech_segment, self._keep_transcription_text)
        specgram = self._normalizer.apply(specgram)
        specgram = self._augmentation_pipeline.transform_feature(specgram)
        return specgram, transcript_part

    def batch_reader_creator(self,
                             manifest_path,
                             batch_size,
                             padding_to=-1,
                             flatten=False,
                             shuffle_method="batch_shuffle"):
        """
        Batch data reader creator for audio data. Return a callable generator
        function to produce batches of data.

        Audio features within one batch will be padded with zeros to have the
        same shape, or a user-defined shape.

        :param manifest_path: Filepath of manifest for audio files.
        :type manifest_path: str
        :param batch_size: Number of instances in a batch.
        :type batch_size: int
        :param padding_to:  If set -1, the maximun shape in the batch
                            will be used as the target shape for padding.
                            Otherwise, `padding_to` will be the target shape.
        :type padding_to: int
        :param flatten: If set True, audio features will be flatten to 1darray.
        :type flatten: bool
        :param shuffle_method: Shuffle method. Options:
                                '' or None: no shuffle.
                                'instance_shuffle': instance-wise shuffle.
                                'batch_shuffle': similarly-sized instances are
                                                 put into batches, and then
                                                 batch-wise shuffle the batches.
                                                 For more details, please see
                                                 ``_batch_shuffle.__doc__``.
                                'batch_shuffle_clipped': 'batch_shuffle' with
                                                         head shift and tail
                                                         clipping. For more
                                                         details, please see
                                                         ``_batch_shuffle``.
                              If sortagrad is True, shuffle is disabled
                              for the first epoch.
        :type shuffle_method: None|str
        :return: Batch reader function, producing batches of data when called.
        :rtype: callable
        """

        def batch_reader():
            # ??????????????????
            manifest = read_manifest(manifest_path=manifest_path,
                                     max_duration=self._max_duration,
                                     min_duration=self._min_duration)
            # ??????????????????????????????
            if self.epoch == 0:
                manifest.sort(key=lambda x: x["duration"], reverse=False)
            else:
                if shuffle_method == "batch_shuffle":
                    manifest = self._batch_shuffle(manifest, batch_size, clipped=False)
                elif shuffle_method == "batch_shuffle_clipped":
                    manifest = self._batch_shuffle(manifest, batch_size, clipped=True)
                elif shuffle_method == "instance_shuffle":
                    self._rng.shuffle(manifest)
                elif shuffle_method is None:
                    pass
                else:
                    raise ValueError("Unknown shuffle method %s." % shuffle_method)
            # ??????????????????
            batch = []
            instance_reader = self._instance_reader_creator(manifest)

            for instance in instance_reader():
                batch.append(instance)
                if len(batch) == batch_size:
                    yield self._padding_batch(batch, padding_to, flatten)
                    batch = []
            if len(batch) >= 1:
                yield self._padding_batch(batch, padding_to, flatten)
            self.epoch += 1

        return batch_reader

    @property
    def feeding(self):
        """????????????????????????exe????????????

        :return: ??????????????????
        :rtype: dict
        """
        feeding_dict = {"audio_spectrogram": 0, "transcript_text": 1}
        return feeding_dict

    @property
    def vocab_size(self):
        """?????????????????????

        :return: ???????????????
        :rtype: int
        """
        return self._speech_featurizer.vocab_size

    @property
    def vocab_list(self):
        """?????????????????????

        :return: ???????????????
        :rtype: list
        """
        return self._speech_featurizer.vocab_list

    def _instance_reader_creator(self, manifest):
        """
        ???????????????????????????reader

        Instance: ??????????????????????????????????????????????????????????????????????????????????????????????????????ID
        """

        def reader():
            for instance in manifest:
                inst = self.process_utterance(instance["audio_filepath"], instance["text"])
                yield inst

        return reader

    def _padding_batch(self, batch, padding_to=-1, flatten=False):
        """
        ????????????????????????????????????????????????batch?????????????????????(??????????????????????????????)

        ??????padding_to???-1????????????????????????????????????????????? ??????????????????????????????
        ?????????' padding_to '??????????????????(???????????????)???

        ?????????flatten??????True???????????????flatten???????????????
        """
        # ??????????????????
        max_length = max([audio.shape[1] for audio, text in batch])
        if padding_to != -1:
            if padding_to < max_length:
                raise ValueError("??????padding_to??????-1???????????????????????????????????????????????????")
            max_length = padding_to
        # ????????????
        padded_audios = []
        texts, text_lens = [], []
        audio_lens = []
        masks = []
        for audio, text in batch:
            padded_audio = np.zeros([audio.shape[0], max_length])
            padded_audio[:, :audio.shape[1]] = audio
            if flatten:
                padded_audio = padded_audio.flatten()
            padded_audios.append(padded_audio)
            if self._is_training:
                texts += text
            else:
                texts.append(text)
            text_lens.append(len(text))
            audio_lens.append(audio.shape[1])
            mask_shape0 = (audio.shape[0] - 1) // 2 + 1
            mask_shape1 = (audio.shape[1] - 1) // 3 + 1
            mask_max_len = (max_length - 1) // 3 + 1
            mask_ones = np.ones((mask_shape0, mask_shape1))
            mask_zeros = np.zeros((mask_shape0, mask_max_len - mask_shape1))
            mask = np.repeat(
                np.reshape(np.concatenate((mask_ones, mask_zeros), axis=1),
                           (1, mask_shape0, mask_max_len)), 32, axis=0)
            masks.append(mask)
        padded_audios = np.array(padded_audios).astype('float32')
        if self._is_training:
            texts = np.expand_dims(np.array(texts).astype('int32'), axis=-1)
            texts = fluid.create_lod_tensor(texts, recursive_seq_lens=[text_lens], place=self._place)
        audio_lens = np.array(audio_lens).astype('int64').reshape([-1, 1])
        masks = np.array(masks).astype('float32')
        return padded_audios, texts, audio_lens, masks

    def _batch_shuffle(self, manifest, batch_size, clipped=False):
        """????????????????????????????????????????????????????????????????????????????????????

        1. ??????????????????????????????????????????
        2. ?????????????????????k??? k?????????[0,batch_size)
        3. ????????????k?????????????????????epoch???????????????????????????
        4. ??????minibatches.

        :param manifest: ????????????
        :type manifest: list
        :param batch_size: ???????????????????????????????????????????????????????????????????????????
        :type batch_size: int
        :param clipped: ??????????????????(?????????)?????????(??????????????????)?????????
        :type clipped: bool
        :return: Batch shuffled mainifest.
        :rtype: list
        """
        manifest.sort(key=lambda x: x["duration"])
        shift_len = self._rng.randint(0, batch_size - 1)
        batch_manifest = list(zip(*[iter(manifest[shift_len:])] * batch_size))
        self._rng.shuffle(batch_manifest)
        batch_manifest = [item for batch in batch_manifest for item in batch]
        if not clipped:
            res_len = len(manifest) - shift_len - len(batch_manifest)
            batch_manifest.extend(manifest[-res_len:])
            batch_manifest.extend(manifest[0:shift_len])
        return batch_manifest
