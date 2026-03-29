# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from typing import Any

import tqdm
import tqdm.auto
from rich.progress import Progress, TaskID


# A class that provides the same interface as tqdm,
# but displays progress bars using Rich.
class TqdmShim(tqdm.tqdm):
    def __init__(self, *args: Any, **kwargs: Any):
        # Pre-initialize so display()/close() are safe if called during super().__init__().
        self.rich_progress: Progress | None = None
        self.rich_task_id: TaskID | None = None

        # Let tqdm initialize first so self.disable and self.leave are set.
        super().__init__(*args, **kwargs)

        if not self.disable:
            self.rich_progress = Progress(transient=not self.leave)
            self.rich_progress.start()
            self.rich_task_id = self.rich_progress.add_task(
                self.desc or "",
                total=self.total,
            )

    def display(self, *args: Any, **kwargs: Any):
        if self.rich_progress is not None and self.rich_task_id is not None:
            self.rich_progress.update(
                self.rich_task_id,
                description=self.desc,
                total=self.total,
                completed=self.n,
            )

    def close(self, *args: Any, **kwargs: Any):
        if self.rich_progress is not None:
            self.rich_progress.stop()
            self.rich_progress = None
        super().close()


def patch_tqdm():
    tqdm.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
    tqdm.auto.tqdm = TqdmShim  # ty:ignore[invalid-assignment]
