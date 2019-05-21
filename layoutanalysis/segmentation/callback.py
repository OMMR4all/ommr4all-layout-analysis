from abc import ABC, abstractmethod


class SegmentationCallback:
    def __init__(self, total_steps: int = 15, total_pages: int = 1):
        super().__init__()
        self.__total_pages = total_pages
        self.__state = 0
        self.__page_state = 0
        self.__total_steps = total_steps

    def get_progress(self):
        return self.__state / self.__total_pages

    def get_current_page_progress(self):
        return self.__page_state / self.__total_steps

    def update_current_page_state(self, state=None):
        if state is not None:
            self.__page_state = 0
        else:
            self.__page_state += 1
        self.changed()

    def update_total_state(self):
        self.__state += 1
        self.changed()

    def reset_page_state(self):
        self.__state = 0
        self.changed()

    def reset_total_state(self):
        self.__page_state = 0
        self.changed()

    @abstractmethod
    def changed(self):
        pass


class SegmentationDummyCallback(SegmentationCallback):
    def changed(self):
        print("Total progress: {} Page progress: {}".format(self.get_progress(), self.get_current_page_progress()))
        pass

    def __init__(self, total_steps: int = 15, total_pages: int = 1):
        super().__init__(total_steps, total_pages)

