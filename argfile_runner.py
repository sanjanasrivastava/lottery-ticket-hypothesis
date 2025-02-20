# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a runner for each set of flags generated by an argfile."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import fire


def run(runner, argfile):
  flags = subprocess.check_output(['python', argfile])
  # for flag in flags.split('\n'):
  for flag in flags.decode('utf-8').split('\n'):
    subprocess.call(['python', runner, flag])


# def main():
def main(unused_argv):
  fire.Fire(run)

if __name__ == '__main__':
  # main()
  main(0)
