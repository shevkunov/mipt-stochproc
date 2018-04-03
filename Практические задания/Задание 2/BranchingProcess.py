class Person:
    def __init__(self, description):
        description_split = description.split('\t')
        self.name = description_split[0]
        self.gender = description_split[1]
        self.birthday = description_split[2]
        self.deathdate = description_split[3]
        self.parents = description_split[4].split(';')
        self.siblings = description_split[5].split(';')
        self.spouses = description_split[6].split(';')
        self.children = description_split[7].split(';')

    def __str__(self):
        line = self.name + '\t'
        line += self.gender + '\t'
        line += self.birthday + '\t'
        line += self.deathdate + '\t'
        line += ';'.join(self.parents) + '\t'
        line += ';'.join(self.siblings) + '\t'
        line += ';'.join(self.spouses) + '\t'
        line += ';'.join(self.children)
        return line


class BranchingProcess:
    def __init__(self):
        self.generations = []

    def __str__(self):
        line = ''
        for i in range(len(self.generations)):
            line += str(i) + ':\n'
            for j in self.generations[i]:
                line += j.__str__() + '\n'
        return line


def read_from_files(files):
    processes = []

    for file_path in files:
        processes.append(BranchingProcess())
        with open(file_path) as file_in:
            for line in file_in:
                if line == '\n':
                    processes.append(BranchingProcess())
                else:
                    i = line.index('\t')
                    generation = int(line[:i])
                    if len(processes[-1].generations) - 1 < generation:
                        processes[-1].generations.append([])

                    processes[-1].generations[-1].append(Person(line[i+1:-1]))

    return processes[:-1]


if __name__ == '__main__':
    processes = read_from_files(['B.txt'])
    for pr in processes:
        print(pr)