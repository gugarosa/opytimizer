from opytimizer.utils import callback


def test_callback():
    new_callback = callback.Callback()

    new_callback.on_task_begin(None)
    new_callback.on_task_end(None)
    new_callback.on_iteration_begin(None, None)
    new_callback.on_iteration_end(None, None)
    new_callback.on_evaluate_before()
    new_callback.on_evaluate_after()
    new_callback.on_update_before()
    new_callback.on_update_after()


def test_callback_vessel():
    new_callback_1 = callback.Callback()
    new_callback_2 = callback.Callback()

    new_callback_vessel = callback.CallbackVessel(
        [new_callback_1, new_callback_2])

    new_callback_vessel.on_task_begin(None)
    new_callback_vessel.on_task_end(None)
    new_callback_vessel.on_iteration_begin(None, None)
    new_callback_vessel.on_iteration_end(None, None)
    new_callback_vessel.on_evaluate_before()
    new_callback_vessel.on_evaluate_after()
    new_callback_vessel.on_update_before()
    new_callback_vessel.on_update_after()


def test_callback_vessel_callbacks():
    new_callback_vessel = callback.CallbackVessel([])

    assert new_callback_vessel.callbacks == []


def test_callback_vessel_callbacks_setter():
    new_callback_vessel = callback.CallbackVessel([])

    try:
        new_callback_vessel.callbacks = 1
    except:
        new_callback_vessel.callbacks = []

    assert new_callback_vessel.callbacks == []


def test_checkpoint_callback():
    new_checkpoint_callback = callback.CheckpointCallback()

    assert new_checkpoint_callback.file_path == 'checkpoint.pkl'
    assert new_checkpoint_callback.frequency == 0


def test_checkpoint_callback_file_path_setter():
    new_checkpoint_callback = callback.CheckpointCallback()

    try:
        new_checkpoint_callback.file_path = 1
    except:
        new_checkpoint_callback.file_path = 'out'

    assert new_checkpoint_callback.file_path == 'out'


def test_checkpoint_callback_frequency_setter():
    new_checkpoint_callback = callback.CheckpointCallback()

    try:
        new_checkpoint_callback.frequency = 'a'
    except:
        new_checkpoint_callback.frequency = 1

    assert new_checkpoint_callback.frequency == 1

    try:
        new_checkpoint_callback.frequency = -1
    except:
        new_checkpoint_callback.frequency = 1

    assert new_checkpoint_callback.frequency == 1


def test_checkpoint_callback_on_iteration_end():
    new_checkpoint_callback = callback.CheckpointCallback(frequency=1)

    class Model:
        def save(self, file_path):
            pass

    model = Model()

    new_checkpoint_callback.on_iteration_end(1, model)
