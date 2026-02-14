# Make sure there is a path, from home to goal (food) ..
#
# Get an environment where there is a path from home-base to food
# (traget location).  Try several times (max_tries). It is possible
# that the barrier density is so high that no path is found even after
# several environment creations (if so, give up!).
def get_env_with_a_path(env, max_tries=50, pit=0):
    path, tries = None, 0
    x1, y1 = env.get_agent_location()
    while path is None:  # not there_is_a_path(env):
        tries += 1
        target = env.get_closest_food()[0]
        path = plan_a_path(target, x1, y1, env)
        if path is None:
            if pit:
                print('\n# NOTE: changing env to get a path ..\n')
            foods=env.get_closest_food() # at least one food
            assert len(foods) == 1
            food_loc=foods[0] # at least one food
            # can change map here (this is a stub in base class)
            env.change_env_map(change_type=2, food_loc=food_loc)
            
        if tries >= max_tries:
            if pit:
                print(
                    '\n# Tried too many times: %d .. couldnt get a good enviromment..\n'
                    % tries
                )
            return None
    return env


#####


# plan and return a path from x1, y1, to target, given envrionment env.
# The returned structure will be a map: loc -> action .
#
# NOTE: may return NONE if no path is found.
def plan_a_path(target, x1, y1, env, max_multiple=5, pit=0):
    x2, y2 = target  # goal
    dim = env.get_dimension()
    # print('# in path planning, target: ', target)
    # Put start (x1, y1) in queue
    #
    # Repeat: Take 1st node out from queue. for each neib, if not
    # yet visited, insert in the queue with an estimate to goal,
    # sort queue by estimate to goal.
    queue, visited, last = [((x1, y1), 0, None, None)], set(), None
    parent = {}  # Allows us to construct the path
    itr = 0
    # If a node is too far from target, as a multiple of straight grid
    # path length from to goal, don't add it to queue.  (This way, we
    # do not need to know the dimensions of the grid.. )
    max_dist = max(max_multiple * (abs(x1 - x2) + abs(y1 - y2)), 2 * dim)
    max_dist_found = 0
    while queue != []:
        itr += 1
        node = queue.pop()  # the last is shortest estimate
        node_xy = node[0]
        dist_to_src = node[1]
        if 0 and queue != []:
            print(
                '# dist:',
                node[1],
                ', farthest:',
                queue[0][1],
                'next closest:',
                queue[-1][1],
            )
        visited.add(node_xy)
        parent[node_xy] = node[3]
        neibs = get_non_barrier_neibs(env, node_xy)

        for n in neibs:
            if n == (x2, y2):  # done: reached the target
                last = node_xy
                break
            # Dont add children (nodes) that are too far ..
            # ( it's an indication that you are going out of
            #   the environment )
            # estimated (lower-bound) distance to goal
            dist_to_goal = abs(n[0] - x2) + abs(n[1] - y2)
            if dist_to_goal > max_dist:
                max_dist_found = max(dist_to_goal, max_dist_found)
                continue
            if n not in visited:
                visited.add(n)
                queue.append((n, dist_to_src+1, dist_to_goal, node_xy))
        if last:
            break
        # Shortest estimated distance last
        queue.sort( key=lambda x: -( x[1]+x[2] ) )
        # print('# queue is now:', queue)
        if itr % 100 == 99 and pit:
            print('# .. continuing the loop:', itr, len(queue))
            print('# .. target is:', x2, y2)
            if queue != []:
                print('# .. last queue entry:', queue[-1])

    # print('\n# itrs in planning:', itr)
    if last is None:  # no path was found
        if pit:
            print('\n# (In ENV planning) did NOT find a path.. (from ',
                  x1, y1, 'to', x2, y2, ')')
            print('# max dist found (the node may been skipped): ', max_dist_found )
        return None

    # String/costruct the path, meaning the action to take at each
    # location, on the path from start to finish
    #
    # last is the node right before goal
    node_xy, planned_move, plan = last, {}, 0
    while node_xy is not None:
        planned_move[node_xy] = get_move_from_to(node_xy, (x2, y2))
        x2, y2 = node_xy
        node_xy = parent.get(node_xy, None)
        plan += 1

    if pit:
        print('# found a path of length: ', plan, 'from', x1, y1, 'to', target)
    return planned_move


def get_non_barrier_neibs(env, node):
    x, y = node
    neibs = []
    if not env.is_barrier_at_loc(x + 1, y):
        neibs.append((x + 1, y))
    if not env.is_barrier_at_loc(x, y + 1):
        neibs.append((x, y + 1))
    if not env.is_barrier_at_loc(x - 1, y):
        neibs.append((x - 1, y))
    if not env.is_barrier_at_loc(x, y - 1):
        neibs.append((x, y - 1))
    return neibs


# Assumes n1 and n2 are adjacent
def get_move_from_to(n1, n2):
    x1, y1 = n1
    x2, y2 = n2
    if y1 == y2:
        if x1 == x2 - 1:  # mv fwd (increase x)
            return 0
        if x1 == x2 + 1:  # mv backward
            return 2
        assert False  # should not happen
    assert x1 == x2
    if y1 == y2 - 1:  # mv up in y
        return 1
    if y1 == y2 + 1:  # mv down
        return 3
    assert False


####
