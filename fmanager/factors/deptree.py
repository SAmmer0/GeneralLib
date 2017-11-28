#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-28 10:59:42
# @Author  : Hao Li (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
该模块专门用于解析各个因子之间的依赖关系
'''

from fmanager.factors.dictionary import get_factor_dict


class Node(object):
    '''
    节点类型，用于表示因子之数据计算间的依赖关系
    因子之间的依赖是一种特殊的树结构
    一个节点可以依赖于多个（或者0个）其他的节点，同时一个节点也可以被其他多个节点依赖
    为此，每个节点维护它所依赖的项（把这些项定义为该节点的子节点）以及父节点
    '''

    def __init__(self, name, dep=None, parent=None):
        '''
        Parameter
        ---------
        name: string
            因子的名称
        dep: iterable, default None
            依赖项，默认为None，表示没有依赖项，或者后续添加
        parent: iterable, default None
            被依赖项，默认为None
        '''
        self.name = name
        if dep is not None:
            self._descendants = dep
        else:
            self._descendants = []
        if parent is not None:
            self._parent = parent
        else:
            self._parent = []

    def has_descendant(self, other_node):
        '''
        判断给定的节点是否是该节点的子节点（依赖项）

        Parameter
        ---------
        other_node: Node
            需要判断的节点

        Return
        ------
        out: boolean
            如果other_node是该节点的“后代”（直接后代或者间接后代），则返回True，其他返回False
        '''
        if other_node in self._descendants:
            return True
        else:
            for desc in self._descendants:
                if desc.has_descendant(other_node):
                    return True
            return False

    def has_descendant_str(self, other_node_str):
        '''
        判断给定的名称的节点是否是该节点的子节点（依赖项）

        Parameter
        ---------
        other_node: string
            需要判断的节点名称

        Retuen
        ------
        out: boolean
            如果该节点包含名称为other_node_str“后代”，则返回True，其他返回False
        '''
        other_node = Node(other_node_str)
        return self.has_descendant(other_node)

    def add_parent(self, parent):
        '''
        向该节点中添加父节点

        Parameter
        ---------
        parent: Node
            添加的父节点
        '''
        self._parent.append(parent)

    def add_descendant(self, descendant):
        '''
        向该节点中添加子节点

        Parameter
        ---------
        descendant: Node
            需要添加的子节点
        '''
        descendant.add_parent(self)
        self._descendants.append(descendant)

    @property
    def descendants(self):
        '''
        只读属性，用于读取当前节点的子节点，不能用于修改子节点
        '''
        return self._descendants

    def __eq__(self, other):
        '''
        Parameter
        ---------
        other: Node
            通过节点的名称判断两个节点是否相同
        '''
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        cls_name = type(self).__name__
        return '<{} name={!r}>'.format(cls_name, self.name)

    def __str__(self):
        cls_name = type(self).__name__
        return '{}(name={!r}, descendants={!r})'.format(cls_name, self.name, self._descendants)


def build_dependency_tree(fd=None):
    '''
    每一个因子都是一个tree的根节点，如果有依赖项，则会存储依赖项的引用

    Parameter
    ---------
    fd: dict, default None
        因子字典，默认为None表示从模块中自动获取因子字典

    Return
    ------
    out: list
        结果列表中存储所有的因子，每个因子以Node类型表示，并且如果有依赖项，则在该节点内会指向同样在
        结果列表中的依赖项节点
    '''
    if fd is None:
        fd = get_factor_dict()
    nodes = {f: Node(f) for f in fd}  # 所有节点初始化，此时不包含任何依赖关系
    for f in nodes:
        dependencys = fd[f]['factor'].dependency
        cur_node = nodes[f]
        if dependencys is not None:
            for dep in dependencys:
                cur_node.add_descendant(nodes[dep])
    return sorted(nodes.values(), key=lambda x: x.name)


def dependency_order(tree=None):
    '''
    通过给定的依赖树来生成依赖的顺序，生成的最终顺序要求任何一个节点一定在被依赖的节点之后

    Parameter
    ---------
    tree: list like, default None
        依赖树，列表内的元素为因子节点

    Return
    ------
    out: list
        依赖顺序列表，顺序按照上述描述来排列，元素为因子节点
    '''
    if tree is None:
        tree = build_dependency_tree()
    out = []

    def add_node(node, container):
        '''
        通过递归的方法将当前节点添加到容器中，如果当前节点的依赖节点不在容器中，则先添加依赖节点
        '''
        if node not in container:
            for dep in node.descendants:
                add_node(dep, container)
            container.append(node)

    for node in tree:
        add_node(node, out)
    return out


if __name__ == '__main__':
    dep_tree = build_dependency_tree()
    fd = get_factor_dict()
    for node in dep_tree:
        dep = sorted(node._descendants, key=lambda x: x.name)
        fd_dep = fd[node.name]['factor'].dependency
        if fd_dep is None:
            fd_dep = []
        else:
            fd_dep = sorted([Node(f) for f in fd_dep], key=lambda x: x.name)
        assert fd_dep == dep
    print('test passes')
    order = dependency_order(dep_tree)
